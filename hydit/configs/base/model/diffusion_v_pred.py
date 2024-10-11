# =========================
#   Diffusion Config
# =========================

learn_sigma = True              # Learn extra channels for sigma.
predict_type = 'v_prediction'   # Diffusion predict type, choices=list(PREDICT_TYPE)
noise_schedule = 'scaled_linear'# Noise schedule,  choices=list(NOISE_SCHEDULES)
beta_start = 0.00085            # Beta start value
beta_end = 0.02                 # Beta end value
sigma_small = False
mse_loss_weight_type = 'constant' # Min_SNR_gamma. Can be constant or min_snr_<gamma> where gamma is a integer. 5 is recommended in the paper.
model_var_type = None           # Specify the model variable type.
noise_offset = 0.0              # Add extra noise to the input image.