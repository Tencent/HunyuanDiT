
import torch
import gradio as gr
from transformers import T5EncoderModel, MT5Tokenizer, BertModel, BertTokenizer
from diffusers import AutoencoderKL, DDPMScheduler
from modules import prompt_parser, shared, rng, devices, processing, scripts, masking, sd_models, sd_samplers_common, images, paths, face_restoration, script_callbacks
from modules.sd_hijack import model_hijack
from modules.timer import Timer
from hunyuan_utils.utils import dit_sampler_dict, hunyuan_transformer_config_v12, retrieve_timesteps, get_timesteps, get_resize_crop_region_for_grid, unload_model, prepare_extra_step_kwargs, prepare_latents_txt2img, prepare_latents_img2img, guess_dit_model, convert_hunyuan_to_diffusers 
from hunyuan_utils import sd_hijack_clip_diffusers, diffusers_learned_conditioning
import os
import numpy as np
from PIL import Image, ImageOps
import cv2
import hashlib

shared.clip_l_model = None
shared.mt5_model = None
shared.vae_model = None

def sample_txt2img(self, conditioning, unconditional_conditioning, seeds):
    # define sampler""
    self.sampler = dit_sampler_dict.get((self.sampler_name+" "+self.scheduler.replace("Automatic","")).strip(),DDPMScheduler()).from_pretrained(shared.opts.Hunyuan_model_path,subfolder="scheduler")
    # reuse webui generated conditionings
    _, tensor = prompt_parser.reconstruct_multicond_batch(conditioning, 0)
    prompt_embeds = tensor["crossattn"]
    prompt_attention_mask = tensor["mask"]
    prompt_embeds_2 = tensor["crossattn_2"]
    prompt_attention_mask_2 = tensor["mask_2"]
    uncond = prompt_parser.reconstruct_cond_batch(unconditional_conditioning, 0)
    negative_prompt_embeds = uncond["crossattn"]
    negative_prompt_attention_mask = uncond["mask"]
    negative_prompt_embeds_2 = uncond["crossattn_2"]
    negative_prompt_attention_mask_2 = uncond["mask_2"]
    # 4. Prepare timesteps
    self.sampler.set_timesteps(self.steps, device=devices.device)
    timesteps = self.sampler.timesteps
    shared.state.sampling_steps = len(timesteps)
    # 5. Prepare latents.
    latent_channels = self.sd_model.config.in_channels
    generators = [rng.create_generator(seed) for seed in seeds]
    latents = prepare_latents_txt2img(
        2 ** (len(shared.vae_model.config.block_out_channels) - 1),
        self.sampler,
        self.batch_size,
        latent_channels,
        self.height,
        self.width,
        prompt_embeds.dtype,
        torch.device("cuda") if shared.opts.randn_source == "GPU" else torch.device("cpu"),
        generators,
        None
    ).to(devices.device)
    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = prepare_extra_step_kwargs(self.sampler, generators, 0.0)

    # 7 create image_rotary_emb, style embedding & time ids
    grid_height = self.height // 8 // self.sd_model.config.patch_size
    grid_width = self.width // 8 // self.sd_model.config.patch_size
    base_size = 512 // 8 // self.sd_model.config.patch_size
    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size)
    from diffusers.models.embeddings import get_2d_rotary_pos_embed
    image_rotary_emb = get_2d_rotary_pos_embed(
        self.sd_model.inner_dim // self.sd_model.num_heads, grid_crops_coords, (grid_height, grid_width)
    )
    style = torch.tensor([0], device=devices.device)

    target_size = (self.height, self.width)
    add_time_ids = list((1024, 1024) + target_size + (0,0))
    add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype)
    if self.cfg_scale > 1:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask])
        prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
        prompt_attention_mask_2 = torch.cat([negative_prompt_attention_mask_2, prompt_attention_mask_2])
        add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
        style = torch.cat([style] * 2, dim=0)
    add_time_ids = add_time_ids.to(dtype=prompt_embeds.dtype, device=devices.device).repeat(
        self.batch_size, 1
    )
    style = style.to(device=devices.device).repeat(self.batch_size)
    for i, t in enumerate(timesteps):
        if shared.state.interrupted or shared.state.skipped:
            raise sd_samplers_common.InterruptedException
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if self.cfg_scale > 1.0 else latents
        latent_model_input = self.sampler.scale_model_input(latent_model_input, t)

        # expand scalar t to 1-D tensor to match the 1st dim of latent_model_input
        t_expand = torch.tensor([t] * latent_model_input.shape[0], device=devices.device).to(
            dtype=latent_model_input.dtype
        )
        # predict the noise residual
        noise_pred = self.sd_model(
            latent_model_input,
            t_expand,
            encoder_hidden_states=prompt_embeds,
            text_embedding_mask=prompt_attention_mask,
            encoder_hidden_states_t5=prompt_embeds_2,
            text_embedding_mask_t5=prompt_attention_mask_2,
            image_meta_size=add_time_ids,
            style=style,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0]

        noise_pred, _ = noise_pred.chunk(2, dim=1)

        # perform guidance
        if self.cfg_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = self.sampler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
        # update process
        shared.state.sampling_step += 1
        shared.total_tqdm.update()
    return latents.to(devices.dtype)

def sample_img2img(self, conditioning, unconditional_conditioning, seeds):
    # define sampler
    self.sampler = dit_sampler_dict.get((self.sampler_name+" "+self.scheduler.replace("Automatic","")).strip(),DDPMScheduler()).from_pretrained(shared.opts.Hunyuan_model_path,subfolder="scheduler")
    # reuse webui generated conditionings
    _, tensor = prompt_parser.reconstruct_multicond_batch(conditioning, 0)
    prompt_embeds = tensor["crossattn"]
    prompt_attention_mask = tensor["mask"]
    prompt_embeds_2 = tensor["crossattn_2"]
    prompt_attention_mask_2 = tensor["mask_2"]
    uncond = prompt_parser.reconstruct_cond_batch(unconditional_conditioning, 0)
    negative_prompt_embeds = uncond["crossattn"]
    negative_prompt_attention_mask = uncond["mask"]
    negative_prompt_embeds_2 = uncond["crossattn_2"]
    negative_prompt_attention_mask_2 = uncond["mask_2"]
    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        self.sampler, self.steps, devices.device, None, None
    )
    timesteps, num_inference_steps = get_timesteps(
        self.sampler,
        num_inference_steps,
        self.denoising_strength,
        devices.device,
        denoising_start=None,
    )
    latent_timestep = timesteps[:1].repeat(self.batch_size)
    shared.state.sampling_steps = len(timesteps)
    # 5. Prepare latents.
    latent_channels = self.sd_model.config.in_channels
    latents_outputs = prepare_latents_img2img(
        2 ** (len(shared.vae_model.config.block_out_channels) - 1),
        self.sampler,
        self.image,
        self.batch_size,
        latent_channels,
        self.height,
        self.width,
        prompt_embeds.dtype,
        torch.device("cuda") if shared.opts.randn_source == "GPU" else torch.device("cpu"),
        None,
        seeds,
        latent_timestep
    )
    latents, noise, image_latents = latents_outputs
    self.init_latent = latents
    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = prepare_extra_step_kwargs(self.sampler, None, 0.0)

    # 7 create image_rotary_emb, style embedding & time ids
    grid_height = self.height // 8 // self.sd_model.config.patch_size
    grid_width = self.width // 8 // self.sd_model.config.patch_size
    base_size = 512 // 8 // self.sd_model.config.patch_size
    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size)
    from diffusers.models.embeddings import get_2d_rotary_pos_embed
    image_rotary_emb = get_2d_rotary_pos_embed(
        self.sd_model.inner_dim // self.sd_model.num_heads, grid_crops_coords, (grid_height, grid_width)
    )
    style = torch.tensor([0], device=devices.device)

    target_size = (self.height, self.width)
    add_time_ids = list((1024, 1024) + target_size + (0,0))
    add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype)
    if self.cfg_scale > 1:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask])
        prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
        prompt_attention_mask_2 = torch.cat([negative_prompt_attention_mask_2, prompt_attention_mask_2])
        add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
        style = torch.cat([style] * 2, dim=0)
    add_time_ids = add_time_ids.to(dtype=prompt_embeds.dtype, device=devices.device).repeat(
        self.batch_size, 1
    )
    style = style.to(device=devices.device).repeat(self.batch_size)
    for i, t in enumerate(timesteps):
        if shared.state.interrupted or shared.state.skipped:
            raise sd_samplers_common.InterruptedException
        # expand the latents if we are doing classifier free guidance
        latent_model_input = latents
        latent_model_input = self.sampler.scale_model_input(latent_model_input, t)

        # expand scalar t to 1-D tensor to match the 1st dim of latent_model_input
        t_expand = torch.tensor([t] * latent_model_input.shape[0], device=devices.device).to(
            dtype=latent_model_input.dtype
        )

        # predict the noise residual
        noise_pred = self.sd_model(
            latent_model_input,
            t_expand,
            encoder_hidden_states=prompt_embeds,
            text_embedding_mask=prompt_attention_mask,
            encoder_hidden_states_t5=prompt_embeds_2,
            text_embedding_mask_t5=prompt_attention_mask_2,
            image_meta_size=add_time_ids,
            style=style,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0]

        noise_pred, _ = noise_pred.chunk(2, dim=1)

        # perform guidance
        if self.cfg_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_text - noise_pred_uncond)
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf

        # compute the previous noisy sample x_t -> x_t-1
        latents = self.sampler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
        if latent_channels == 4 and self.image_mask is not None:
            latents = self.mask * self.init_latent + self.nmask * latents
        # update process
        shared.state.sampling_step += 1
        shared.total_tqdm.update()

    return latents.to(devices.dtype)

def init_img2img(self, all_prompts, all_seeds, all_subseeds):
    self.extra_generation_params["Denoising strength"] = self.denoising_strength

    self.image_cfg_scale: float = self.image_cfg_scale if shared.sd_model.cond_stage_key == "edit" else None

    #self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)
    crop_region = None

    image_mask = self.image_mask

    if image_mask is not None:
        # image_mask is passed in as RGBA by Gradio to support alpha masks,
        # but we still want to support binary masks.
        image_mask = processing.create_binary_mask(image_mask, round=self.mask_round)

        if self.inpainting_mask_invert:
            image_mask = ImageOps.invert(image_mask)
            self.extra_generation_params["Mask mode"] = "Inpaint not masked"

        if self.mask_blur_x > 0:
            np_mask = np.array(image_mask)
            kernel_size = 2 * int(2.5 * self.mask_blur_x + 0.5) + 1
            np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), self.mask_blur_x)
            image_mask = Image.fromarray(np_mask)

        if self.mask_blur_y > 0:
            np_mask = np.array(image_mask)
            kernel_size = 2 * int(2.5 * self.mask_blur_y + 0.5) + 1
            np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), self.mask_blur_y)
            image_mask = Image.fromarray(np_mask)

        if self.mask_blur_x > 0 or self.mask_blur_y > 0:
            self.extra_generation_params["Mask blur"] = self.mask_blur

        if self.inpaint_full_res:
            self.mask_for_overlay = image_mask
            mask = image_mask.convert('L')
            crop_region = masking.get_crop_region_v2(mask, self.inpaint_full_res_padding)
            if crop_region:
                crop_region = masking.expand_crop_region(crop_region, self.width, self.height, mask.width, mask.height)
                x1, y1, x2, y2 = crop_region
                mask = mask.crop(crop_region)
                image_mask = images.resize_image(2, mask, self.width, self.height)
                self.paste_to = (x1, y1, x2-x1, y2-y1)
                self.extra_generation_params["Inpaint area"] = "Only masked"
                self.extra_generation_params["Masked area padding"] = self.inpaint_full_res_padding
            else:
                crop_region = None
                image_mask = None
                self.mask_for_overlay = None
                self.inpaint_full_res = False
                massage = 'Unable to perform "Inpaint Only mask" because mask is blank, switch to img2img mode.'
                model_hijack.comments.append(massage)
        else:
            image_mask = images.resize_image(self.resize_mode, image_mask, self.width, self.height)
            np_mask = np.array(image_mask)
            np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
            self.mask_for_overlay = Image.fromarray(np_mask)

        self.overlay_images = []

    latent_mask = self.latent_mask if self.latent_mask is not None else image_mask

    add_color_corrections = shared.opts.img2img_color_correction and self.color_corrections is None
    if add_color_corrections:
        self.color_corrections = []
    imgs = []
    for img in self.init_images:

        # Save init image
        if shared.opts.save_init_img:
            self.init_img_hash = hashlib.md5(img.tobytes()).hexdigest()
            images.save_image(img, path=shared.opts.outdir_init_images, basename=None, forced_filename=self.init_img_hash, save_to_dirs=False, existing_info=img.info)

        image = images.flatten(img, shared.opts.img2img_background_color)

        if crop_region is None and self.resize_mode != 3:
            image = images.resize_image(self.resize_mode, image, self.width, self.height)

        if image_mask is not None:
            if self.mask_for_overlay.size != (image.width, image.height):
                self.mask_for_overlay = images.resize_image(self.resize_mode, self.mask_for_overlay, image.width, image.height)
            image_masked = Image.new('RGBa', (image.width, image.height))
            image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(self.mask_for_overlay.convert('L')))

            self.overlay_images.append(image_masked.convert('RGBA'))

        # crop_region is not None if we are doing inpaint full res
        if crop_region is not None:
            image = image.crop(crop_region)
            image = images.resize_image(2, image, self.width, self.height)

        if image_mask is not None:
            if self.inpainting_fill != 1:
                image = masking.fill(image, latent_mask)

                if self.inpainting_fill == 0:
                    self.extra_generation_params["Masked content"] = 'fill'

        if add_color_corrections:
            self.color_corrections.append(processing.setup_color_correction(image))

        image = np.array(image).astype(np.float32) / 255.0
        image = np.moveaxis(image, 2, 0)

        imgs.append(image)

    if len(imgs) == 1:
        batch_images = np.expand_dims(imgs[0], axis=0).repeat(self.batch_size, axis=0)
        if self.overlay_images is not None:
            self.overlay_images = self.overlay_images * self.batch_size

        if self.color_corrections is not None and len(self.color_corrections) == 1:
            self.color_corrections = self.color_corrections * self.batch_size

    elif len(imgs) <= self.batch_size:
        self.batch_size = len(imgs)
        batch_images = np.array(imgs)
    else:
        raise RuntimeError(f"bad number of images passed: {len(imgs)}; expecting {self.batch_size} or less")

    image = torch.from_numpy(batch_images)
    self.image = image.to(shared.device, dtype=devices.dtype_vae)

def process_images_inner_hunyuan(p: processing.StableDiffusionProcessing) -> processing.Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    if isinstance(p.prompt, list):
        assert(len(p.prompt) > 0)
    else:
        assert p.prompt is not None

    devices.torch_gc()

    seed = processing.get_fixed_seed(p.seed)
    subseed = processing.get_fixed_seed(p.subseed)

    if p.restore_faces is None:
        p.restore_faces = shared.opts.face_restoration

    if p.tiling is None:
        p.tiling = shared.opts.tiling
    
    # disable refiner
    '''
    if p.refiner_checkpoint not in (None, "", "None", "none"):
        p.refiner_checkpoint_info = sd_models.get_closet_checkpoint_match(p.refiner_checkpoint)
        if p.refiner_checkpoint_info is None:
            raise Exception(f'Could not find checkpoint with name {p.refiner_checkpoint}')
    '''
    p.sd_model_name = shared.sd_model.sd_checkpoint_info.name_for_extra
    p.sd_model_hash = shared.sd_model.sd_model_hash
    # disable stable diffusion vae
    '''
    p.sd_vae_name = sd_vae.get_loaded_vae_name()
    p.sd_vae_hash = sd_vae.get_loaded_vae_hash()
    '''
    model_hijack.apply_circular(p.tiling)
    model_hijack.clear_comments()

    p.setup_prompts()

    if isinstance(seed, list):
        p.all_seeds = seed
    else:
        p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]

    if isinstance(subseed, list):
        p.all_subseeds = subseed
    else:
        p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

    if os.path.exists(shared.cmd_opts.embeddings_dir) and not p.do_not_reload_embeddings:
        model_hijack.embedding_db.load_textual_inversion_embeddings()

    if p.scripts is not None:
        p.scripts.process(p)

    infotexts = []
    output_images = []
    with torch.no_grad():
        with devices.autocast():
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

            # disable stable diffusion vae
            '''
            # for OSX, loading the model during sampling changes the generated picture, so it is loaded here
            if shared.opts.live_previews_enable and opts.show_progress_type == "Approx NN":
                sd_vae_approx.model()

            sd_unet.apply_unet()
            '''
        if shared.state.job_count == -1:
            shared.state.job_count = p.n_iter

        for n in range(p.n_iter):
            p.iteration = n

            if shared.state.skipped:
                shared.state.skipped = False

            if shared.state.interrupted or shared.state.stopping_generation:
                break

            sd_models.reload_model_weights()  # model can be changed for example by refiner

            p.prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
            p.subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

            # disable webui rng for stable diffusion
            #p.rng = rng.ImageRNG((opt_C, p.height // opt_f, p.width // opt_f), p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength, seed_resize_from_h=p.seed_resize_from_h, seed_resize_from_w=p.seed_resize_from_w)

            if p.scripts is not None:
                p.scripts.before_process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)

            if len(p.prompts) == 0:
                break
            # disabled sd webui type loras
            '''
            p.parse_extra_network_prompts()

            if not p.disable_extra_networks:
                with devices.autocast():
                    extra_networks.activate(p, p.extra_network_data)
            '''
            if p.scripts is not None:
                p.scripts.process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)

            p.setup_conds()

            # p.extra_generation_params.update(model_hijack.extra_generation_params)

            # params.txt should be saved after scripts.process_batch, since the
            # infotext could be modified by that callback
            # Example: a wildcard processed by process_batch sets an extra model
            # strength, which is saved as "Model Strength: 1.0" in the infotext
            if n == 0 and not shared.cmd_opts.no_prompt_history:
                with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
                    processed = processing.Processed(p, [])
                    file.write(processed.infotext(p, 0))

            for comment in model_hijack.comments:
                p.comment(comment)

            if p.n_iter > 1:
                shared.state.job = f"Batch {n+1} out of {p.n_iter}"

            sd_models.apply_alpha_schedule_override(p.sd_model, p)

            with devices.without_autocast() if devices.unet_needs_upcast else devices.autocast():
                samples_ddim = p.sample(conditioning=p.c, unconditional_conditioning=p.uc, seeds=p.seeds)

            if p.scripts is not None:
                ps = scripts.PostSampleArgs(samples_ddim)
                p.scripts.post_sample(p, ps)
                samples_ddim = ps.samples

            if getattr(samples_ddim, 'already_decoded', False):
                x_samples_ddim = samples_ddim
            else:
                if shared.opts.sd_vae_decode_method != 'Full':
                    p.extra_generation_params['VAE Decoder'] = shared.opts.sd_vae_decode_method
                x_samples_ddim = shared.vae_model.decode(samples_ddim / shared.vae_model.config.scaling_factor, return_dict=False)[0]

            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).float().numpy()

            del samples_ddim

            devices.torch_gc()

            shared.state.nextjob()

            if p.scripts is not None:
                p.scripts.postprocess_batch(p, x_samples_ddim, batch_number=n)

                p.prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
                p.negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]

                batch_params = scripts.PostprocessBatchListArgs(list(x_samples_ddim))
                p.scripts.postprocess_batch_list(p, batch_params, batch_number=n)
                x_samples_ddim = batch_params.images

            def infotext(index=0, use_main_prompt=False):
                return processing.create_infotext(p, p.prompts, p.seeds, p.subseeds, use_main_prompt=use_main_prompt, index=index, all_negative_prompts=p.negative_prompts)

            save_samples = p.save_samples()

            for i, x_sample in enumerate(x_samples_ddim):
                p.batch_index = i

                x_sample = 255. * x_sample
                x_sample = x_sample.astype(np.uint8)

                if p.restore_faces:
                    if save_samples and shared.opts.save_images_before_face_restoration:
                        images.save_image(Image.fromarray(x_sample), p.outpath_samples, "", p.seeds[i], p.prompts[i], shared.opts.samples_format, info=infotext(i), p=p, suffix="-before-face-restoration")

                    devices.torch_gc()

                    x_sample = face_restoration.restore_faces(x_sample)
                    devices.torch_gc()

                image = Image.fromarray(x_sample)

                if p.scripts is not None:
                    pp = scripts.PostprocessImageArgs(image)
                    p.scripts.postprocess_image(p, pp)
                    image = pp.image

                mask_for_overlay = getattr(p, "mask_for_overlay", None)

                if not shared.opts.overlay_inpaint:
                    overlay_image = None
                elif getattr(p, "overlay_images", None) is not None and i < len(p.overlay_images):
                    overlay_image = p.overlay_images[i]
                else:
                    overlay_image = None

                if p.scripts is not None:
                    ppmo = scripts.PostProcessMaskOverlayArgs(i, mask_for_overlay, overlay_image)
                    p.scripts.postprocess_maskoverlay(p, ppmo)
                    mask_for_overlay, overlay_image = ppmo.mask_for_overlay, ppmo.overlay_image

                if p.color_corrections is not None and i < len(p.color_corrections):
                    if save_samples and shared.opts.save_images_before_color_correction:
                        image_without_cc, _ = processing.apply_overlay(image, p.paste_to, overlay_image)
                        images.save_image(image_without_cc, p.outpath_samples, "", p.seeds[i], p.prompts[i], shared.opts.samples_format, info=infotext(i), p=p, suffix="-before-color-correction")
                    image = processing.apply_color_correction(p.color_corrections[i], image)

                # If the intention is to show the output from the model
                # that is being composited over the original image,
                # we need to keep the original image around
                # and use it in the composite step.
                image, original_denoised_image = processing.apply_overlay(image, p.paste_to, overlay_image)

                if p.scripts is not None:
                    pp = scripts.PostprocessImageArgs(image)
                    p.scripts.postprocess_image_after_composite(p, pp)
                    image = pp.image

                if save_samples:
                    images.save_image(image, p.outpath_samples, "", p.seeds[i], p.prompts[i], shared.opts.samples_format, info=infotext(i), p=p)

                text = infotext(i)
                infotexts.append(text)
                if shared.opts.enable_pnginfo:
                    image.info["parameters"] = text
                output_images.append(image)

                if mask_for_overlay is not None:
                    if shared.opts.return_mask or shared.opts.save_mask:
                        image_mask = mask_for_overlay.convert('RGB')
                        if save_samples and shared.opts.save_mask:
                            images.save_image(image_mask, p.outpath_samples, "", p.seeds[i], p.prompts[i], shared.opts.samples_format, info=infotext(i), p=p, suffix="-mask")
                        if shared.opts.return_mask:
                            output_images.append(image_mask)

                    if shared.opts.return_mask_composite or shared.opts.save_mask_composite:
                        image_mask_composite = Image.composite(original_denoised_image.convert('RGBA').convert('RGBa'), Image.new('RGBa', image.size), images.resize_image(2, mask_for_overlay, image.width, image.height).convert('L')).convert('RGBA')
                        if save_samples and shared.opts.save_mask_composite:
                            images.save_image(image_mask_composite, p.outpath_samples, "", p.seeds[i], p.prompts[i], shared.opts.samples_format, info=infotext(i), p=p, suffix="-mask-composite")
                        if shared.opts.return_mask_composite:
                            output_images.append(image_mask_composite)

            del x_samples_ddim

            devices.torch_gc()

        if not infotexts:
            infotexts.append(processing.Processed(p, []).infotext(p, 0))

        p.color_corrections = None

        index_of_first_image = 0
        unwanted_grid_because_of_img_count = len(output_images) < 2 and shared.opts.grid_only_if_multiple
        if (shared.opts.return_grid or shared.opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
            grid = images.image_grid(output_images, p.batch_size)

            if shared.opts.return_grid:
                text = infotext(use_main_prompt=True)
                infotexts.insert(0, text)
                if shared.opts.enable_pnginfo:
                    grid.info["parameters"] = text
                output_images.insert(0, grid)
                index_of_first_image = 1
            if shared.opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", p.all_seeds[0], p.all_prompts[0], shared.opts.grid_format, info=infotext(use_main_prompt=True), short_filename=not shared.opts.grid_extended_filename, p=p, grid=True)

    # disable sd webui type loras
    '''
    if not p.disable_extra_networks and p.extra_network_data:
        extra_networks.deactivate(p, p.extra_network_data)
    '''
    devices.torch_gc()

    res = processing.Processed(
        p,
        images_list=output_images,
        seed=p.all_seeds[0],
        info=infotexts[0],
        subseed=p.all_subseeds[0],
        index_of_first_image=index_of_first_image,
        infotexts=infotexts,
    )

    if p.scripts is not None:
        p.scripts.postprocess(p, res)

    return res

def load_model_hunyuan(checkpoint_info=None, already_loaded_state_dict=None):
    from modules import sd_hijack
    from diffusers import HunyuanDiT2DModel
    checkpoint_info = checkpoint_info or sd_models.select_checkpoint()

    timer = Timer()

    if sd_models.model_data.sd_model:
        sd_models.model_data.sd_model.to("cpu")
        sd_models.model_data.sd_model = None
        devices.torch_gc()

    timer.record("unload existing model")

    if already_loaded_state_dict is not None:
        state_dict = already_loaded_state_dict
    else:
        state_dict = sd_models.get_checkpoint_state_dict(checkpoint_info, timer)

    timer.record("load weights from state dict")

    sd_model = HunyuanDiT2DModel.from_config(hunyuan_transformer_config_v12)
    print("loading hunyuan DiT")
    checkpoint_config = guess_dit_model(state_dict)
    sd_model.used_config = checkpoint_config
    if checkpoint_config == "hunyuan-original":
        state_dict = convert_hunyuan_to_diffusers(state_dict)
    elif "hunyuan" not in checkpoint_config:
        raise ValueError("Found no hunyuan DiT model")
    sd_model.load_state_dict(state_dict, strict=False)
    del state_dict

    print("loading text encoder and vae")
    shared.clip_l_model = BertModel.from_pretrained(shared.opts.Hunyuan_model_path,subfolder="text_encoder",torch_dtype=devices.dtype).to(devices.device)
    shared.mt5_model = T5EncoderModel.from_pretrained(shared.opts.Hunyuan_model_path,subfolder="text_encoder_2",torch_dtype=devices.dtype).to(devices.device)
    shared.clip_l_model.tokenizer = BertTokenizer.from_pretrained(shared.opts.Hunyuan_model_path,subfolder="tokenizer")
    shared.mt5_model.tokenizer = MT5Tokenizer.from_pretrained(shared.opts.Hunyuan_model_path,subfolder="tokenizer_2")
    shared.clip_l_model = sd_hijack_clip_diffusers.FrozenBertEmbedderWithCustomWords(shared.clip_l_model,sd_hijack.model_hijack)
    shared.mt5_model = sd_hijack_clip_diffusers.FrozenT5EmbedderWithCustomWords(shared.mt5_model,sd_hijack.model_hijack)
    shared.clip_l_model.return_masks = True
    shared.mt5_model.return_masks = True
    shared.vae_model = AutoencoderKL.from_pretrained(shared.opts.Hunyuan_model_path,subfolder="vae",torch_dtype=devices.dtype).to(devices.device)
    
    sd_model.to(devices.dtype)
    sd_model.to(devices.device)
    sd_model.eval()
    sd_model_hash = checkpoint_info.calculate_shorthash()
    sd_model.sd_model_hash = sd_model_hash
    sd_model.sd_model_checkpoint = checkpoint_info.filename
    sd_model.sd_checkpoint_info = checkpoint_info
    sd_model.lowvram = False
    sd_model.is_sd1 = False
    sd_model.is_sd2 = False
    sd_model.is_sdxl = False
    sd_model.is_ssd = False
    sd_model.is_sd3 = False
    sd_model.model = None
    sd_model.first_stage_model = None
    sd_model.cond_stage_key = None
    sd_model.cond_stage_model = None
    sd_model.get_learned_conditioning = diffusers_learned_conditioning.get_learned_conditioning_hunyuan
    sd_models.model_data.set_sd_model(sd_model)
    sd_models.model_data.was_loaded_at_least_once = True

    script_callbacks.model_loaded_callback(sd_model)

    timer.record("scripts callbacks")

    print(f"Model loaded in {timer.summary()}.")

    return sd_model

def reload_model_weights_hunyuan(sd_model=None, info=None, forced_reload=False):
    checkpoint_info = info or sd_models.select_checkpoint()

    timer = Timer()

    if not sd_model:
        sd_model = sd_models.model_data.sd_model

    if sd_model is None:  # previous model load failed
        current_checkpoint_info = None
    else:
        current_checkpoint_info = sd_model.sd_checkpoint_info
        if sd_model.sd_model_checkpoint == checkpoint_info.filename and not forced_reload:
            return sd_model

    sd_model.to(devices.dtype)
    sd_model.to(devices.device)
    if not forced_reload and sd_model is not None and sd_model.sd_checkpoint_info.filename == checkpoint_info.filename:
        return sd_model

    if sd_model is not None:
        sd_models.send_model_to_cpu(sd_model)

    state_dict = sd_models.get_checkpoint_state_dict(checkpoint_info, timer)

    checkpoint_config = guess_dit_model(state_dict)
    if checkpoint_config == "hunyuan-original":
        state_dict = convert_hunyuan_to_diffusers(state_dict)
    elif "hunyuan" not in checkpoint_config:
        raise ValueError("Found no hunyuan DiT model")
    timer.record("find config")

    if sd_model is None or checkpoint_config != sd_model.used_config:
        load_model_hunyuan(checkpoint_info, already_loaded_state_dict=state_dict)
        return sd_models.model_data.sd_model
    try:
        sd_model.load_state_dict(state_dict, strict=False)
        del state_dict
        sd_model_hash = checkpoint_info.calculate_shorthash()
        sd_model.sd_model_hash = sd_model_hash
        sd_model.sd_model_checkpoint = checkpoint_info.filename
        sd_model.sd_checkpoint_info = checkpoint_info
        sd_model.lowvram = False
        sd_model.is_sd1 = False
        sd_model.is_sd2 = False
        sd_model.is_sdxl = False
        sd_model.is_ssd = False
        sd_model.is_sd3 = False
        sd_model.model = None
        sd_model.first_stage_model = None
        sd_model.cond_stage_key = None
        sd_model.cond_stage_model = None
    except Exception:
        print("Failed to load checkpoint, restoring previous")
        state_dict = sd_models.get_checkpoint_state_dict(current_checkpoint_info, timer)
        sd_model.load_state_dict(state_dict, strict=False)
        del state_dict
        sd_model_hash = checkpoint_info.calculate_shorthash()
        sd_model.sd_model_hash = sd_model_hash
        sd_model.sd_model_checkpoint = checkpoint_info.filename
        sd_model.sd_checkpoint_info = checkpoint_info
        sd_model.lowvram = False
        sd_model.is_sd1 = False
        sd_model.is_sd2 = False
        sd_model.is_sdxl = False
        sd_model.is_ssd = False
        sd_model.is_sd3 = False
        sd_model.model = None
        sd_model.first_stage_model = None
        sd_model.cond_stage_key = None
        sd_model.cond_stage_model = None
        raise
    finally:
        script_callbacks.model_loaded_callback(sd_model)
        timer.record("script callbacks")

    print(f"Weights loaded in {timer.summary()}.")

    sd_models.model_data.set_sd_model(sd_model)

    return sd_model

class Script(scripts.Script):
    
    def __init__(self):
        super(Script, self).__init__()
    def title(self):
        return 'Hunyuan DiT'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab    = 't2i'  if not is_img2img else 'i2i'
        is_t2i = 'true' if not is_img2img else 'false'
        uid = lambda name: f'MD-{tab}-{name}'

        with gr.Accordion('Hunyuan DiT', open=False):
            with gr.Row(variant='compact') as tab_enable:
                enabled = gr.Checkbox(label='Enable Hunyuan DiT', value=False,  elem_id=uid('enabled'))                 
        enabled.change(
            fn=on_enable_change,
            inputs=[enabled],
            outputs=None
        )
        return [
            enabled
        ]

def on_enable_change(enabled: bool):
    if enabled:
        print("Enable Hunyuan DiT")
        hijack()
    else:
        print("Disable Hunyuan DiT")
        reset()
        shared.clip_l_model = unload_model(shared.clip_l_model)
        shared.mt5_model = unload_model(shared.mt5_model)
        shared.vae_model = unload_model(shared.vae_model)

def reset():
    ''' unhijack inner APIs '''
    if hasattr(processing,"process_images_inner_original"):
        processing.process_images_inner = processing.process_images_inner_original
    if hasattr(processing.StableDiffusionProcessingTxt2Img,"sample_original"):
        processing.StableDiffusionProcessingTxt2Img.sample = processing.StableDiffusionProcessingTxt2Img.sample_original
    if hasattr(processing.StableDiffusionProcessingImg2Img,"sample_original"):
        processing.StableDiffusionProcessingImg2Img.sample = processing.StableDiffusionProcessingImg2Img.sample_original
    if hasattr(sd_models,"load_model_original"):
        sd_models.load_model = sd_models.load_model_original
    if hasattr(sd_models,"reload_model_weights_original"):
        sd_models.reload_model_weights = sd_models.reload_model_weights_original
    if hasattr(processing.StableDiffusionProcessingImg2Img,"init_img2img_original"):
        processing.StableDiffusionProcessingImg2Img.init = processing.StableDiffusionProcessingImg2Img.init_img2img_original

def hijack():
    ''' hijack inner APIs '''
    if not hasattr(processing,"process_images_inner_original"):
        processing.process_images_inner_original = processing.process_images_inner
    if not hasattr(processing.StableDiffusionProcessingTxt2Img,"sample_original"):
        processing.StableDiffusionProcessingTxt2Img.sample_original = processing.StableDiffusionProcessingTxt2Img.sample
    if not hasattr(processing.StableDiffusionProcessingImg2Img,"sample_original"):
        processing.StableDiffusionProcessingImg2Img.sample_original = processing.StableDiffusionProcessingImg2Img.sample
    if not hasattr(sd_models,"load_model_original"):
        sd_models.load_model_original = sd_models.load_model
    if not hasattr(sd_models,"reload_model_weights_original"):
        sd_models.reload_model_weights_original = sd_models.reload_model_weights
    if not hasattr(processing.StableDiffusionProcessingImg2Img,"init_img2img_original"):
        processing.StableDiffusionProcessingImg2Img.init_img2img_original = processing.StableDiffusionProcessingImg2Img.init
    processing.process_images_inner = process_images_inner_hunyuan
    processing.StableDiffusionProcessingTxt2Img.sample = sample_txt2img
    processing.StableDiffusionProcessingImg2Img.sample = sample_img2img
    sd_models.load_model = load_model_hunyuan
    sd_models.reload_model_weights = reload_model_weights_hunyuan
    processing.StableDiffusionProcessingImg2Img.init = init_img2img

def on_ui_settings():

    shared.opts.add_option("Hunyuan_model_path", shared.OptionInfo("./models/hunyuan", "Hunyuan Model Path",section=('hunyuanDiT', "HunyuanDiT")))

script_callbacks.on_ui_settings(on_ui_settings)
