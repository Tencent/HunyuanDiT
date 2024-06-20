# MIT License
# Copyright (c) 2023 AIGText
# https://github.com/AIGText/GlyphControl-release

from PIL import Image, ImageFont, ImageDraw
import random
import numpy as np
import cv2


# resize height to image_height first, then shrink or pad to image_width
def resize_and_pad_image(pil_image, image_size):
    if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        image_width, image_height = image_size
    elif isinstance(image_size, int):
        image_width = image_height = image_size
    else:
        raise ValueError(f"Image size should be int or list/tuple of int not {image_size}")

    while pil_image.size[1] >= 2 * image_height:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_height / pil_image.size[1]
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    # shrink
    if pil_image.size[0] > image_width:
        pil_image = pil_image.resize((image_width, image_height), resample=Image.BICUBIC)

    # padding
    if pil_image.size[0] < image_width:
        img = Image.new(mode="RGBA", size=(image_width, image_height), color=(255, 255, 255, 0))
        width, _ = pil_image.size
        img.paste(pil_image, ((image_width - width) // 2, 0))
        pil_image = img

    return pil_image


def resize_and_pad_image2(pil_image, image_size):
    if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        image_width, image_height = image_size
    elif isinstance(image_size, int):
        image_width = image_height = image_size
    else:
        raise ValueError(f"Image size should be int or list/tuple of int not {image_size}")

    while pil_image.size[1] >= 2 * image_height:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_height / pil_image.size[1]
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    # shrink
    if pil_image.size[0] > image_width:
        pil_image = pil_image.resize((image_width, image_height), resample=Image.BICUBIC)

    # padding
    if pil_image.size[0] < image_width:
        img = Image.new(mode="RGB", size=(image_width, image_height), color="white")
        width, _ = pil_image.size
        img.paste(pil_image, ((image_width - width) // 2, 0))
        pil_image = img

    return pil_image


def draw_visual_text(image_size, bboxes, rendered_txt_values, num_rows_values=None, align="center"):
    # aligns = ["center", "left", "right"]
    """Render text image based on the glyph instructions, i.e., the list of tuples (text, bbox, num_rows).
       Currently we just use Calibri font to render glyph images.
    """
    # print(image_size, bboxes, rendered_txt_values, num_rows_values, align)
    background = Image.new("RGB", image_size, "white")
    font = ImageFont.truetype("simfang.ttf", encoding='utf-8', size=512)
    if num_rows_values is None:
        num_rows_values = [1] * len(rendered_txt_values)

    text_list = []
    for text, bbox, num_rows in zip(rendered_txt_values, bboxes, num_rows_values):

        if len(text) == 0:
            continue

        text = text.strip()
        if num_rows != 1:
            word_tokens = text.split()
            num_tokens = len(word_tokens)
            index_list = range(1, num_tokens + 1)
            if num_tokens > num_rows:
                index_list = random.sample(index_list, num_rows)
                index_list.sort()
            line_list = []
            start_idx = 0
            for index in index_list:
                line_list.append(
                    " ".join(word_tokens
                             [start_idx: index]
                             )
                )
                start_idx = index
            text = "\n".join(line_list)

        if 'ratio' not in bbox or bbox['ratio'] == 0 or bbox['ratio'] < 1e-4:
            image4ratio = Image.new("RGB", (512, 512), "white")
            draw = ImageDraw.Draw(image4ratio)
            _, _, w, h = draw.textbbox(xy=(0, 0), text=text, font=font)
            ratio = w / h
        else:
            ratio = bbox['ratio']

        width = int(bbox['width'] * image_size[1])
        height = int(width / ratio)
        top_left_x = int(bbox['top_left_x'] * image_size[0])
        top_left_y = int(bbox['top_left_y'] * image_size[1])
        yaw = bbox['yaw']

        text_image = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(text_image)
        x, y, w, h = draw.textbbox(xy=(0, 0), text=text, font=font)
        text_image = Image.new("RGBA", (w, h), (255, 255, 255, 0))
        draw = ImageDraw.Draw(text_image)
        draw.text((-x / 2, -y / 2), text, (0, 0, 0, 255), font=font, align=align)

        text_image_ = resize_and_pad_image2(text_image.convert('RGB'), (288, 48))
        # import pdb; pdb.set_trace()
        text_list.append(np.array(text_image_))

        text_image = resize_and_pad_image(text_image, (width, height))
        text_image = text_image.rotate(angle=-yaw, expand=True, fillcolor=(255, 255, 255, 0))
        # image = Image.new("RGB", (w, h), "white")
        # draw = ImageDraw.Draw(image)
        background.paste(text_image, (top_left_x, top_left_y), mask=text_image)

    return background, text_list


# [{'width': 0.1601562201976776, 'ratio': 81.99999451637203, 'yaw': 0.0, 'top_left_x': 0.712890625, 'top_left_y': 0.0},
#  {'width': 0.134765625, 'ratio': 34.5, 'yaw': 0.0, 'top_left_x': 0.4453125, 'top_left_y': 0.0},


def insert_spaces(string, nSpace):
    if nSpace == 0:
        return string
    new_string = ""
    for char in string:
        new_string += char + " " * nSpace
    return new_string[:-nSpace]


def draw_glyph(text, font='simfang.ttf'):
    if isinstance(font, str):
        font = ImageFont.truetype(font, encoding='utf-8', size=512)
    g_size = 50
    W, H = (512, 80)
    new_font = font.font_variant(size=g_size)
    img = Image.new(mode='1', size=(W, H), color=0)
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = new_font.getbbox(text)
    text_width = max(right-left, 5)
    text_height = max(bottom - top, 5)
    ratio = min(W*0.9/text_width, H*0.9/text_height)
    new_font = font.font_variant(size=int(g_size*ratio))

    text_width, text_height = new_font.getsize(text)
    offset_x, offset_y = new_font.getoffset(text)
    x = (img.width - text_width) // 2
    y = (img.height - text_height) // 2 - offset_y//2
    draw.text((x, y), text, font=new_font, fill='white')
    img = np.expand_dims(np.array(img), axis=2).astype(np.float64)

    return img


def draw_glyph2(text, polygon, font='simfang.ttf', vertAng=10, scale=1, width=1024, height=1024, add_space=True):
    if isinstance(font, str):
        font = ImageFont.truetype(font, encoding='utf-8', size=60)
    enlarge_polygon = polygon*scale
    rect = cv2.minAreaRect(enlarge_polygon)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    w, h = rect[1]
    angle = rect[2]
    if angle < -45:
        angle += 90
    angle = -angle
    if w < h:
        angle += 90

    vert = False
    if (abs(angle) % 90 < vertAng or abs(90-abs(angle) % 90) % 90 < vertAng):
        _w = max(box[:, 0]) - min(box[:, 0])
        _h = max(box[:, 1]) - min(box[:, 1])
        if _h >= _w:
            vert = True
            angle = 0

    img = np.zeros((height*scale, width*scale, 3), np.uint8)
    img = Image.fromarray(img)

    # infer font size
    image4ratio = Image.new("RGB", img.size, "white")
    draw = ImageDraw.Draw(image4ratio)
    _, _, _tw, _th = draw.textbbox(xy=(0, 0), text=text, font=font)
    text_w = min(w, h) * (_tw / _th)
    if text_w <= max(w, h):
        # add space
        if len(text) > 1 and not vert and add_space:
            for i in range(1, 100):
                text_space = insert_spaces(text, i)
                _, _, _tw2, _th2 = draw.textbbox(xy=(0, 0), text=text_space, font=font)
                if min(w, h) * (_tw2 / _th2) > max(w, h):
                    break
            text = insert_spaces(text, i-1)
        font_size = min(w, h)*0.80
    else:
        # shrink = 0.75 if vert else 0.85
        shrink = 1.0
        font_size = min(w, h) / (text_w/max(w, h)) * shrink
    new_font = font.font_variant(size=int(font_size))

    left, top, right, bottom = new_font.getbbox(text)
    text_width = right-left
    text_height = bottom - top

    layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if not vert:
        draw.text((rect[0][0]-text_width//2, rect[0][1]-text_height//2-top), text, font=new_font, fill=(255, 255, 255, 255))
    else:
        x_s = min(box[:, 0]) + _w//2 - text_height//2
        y_s = min(box[:, 1])
        for c in text:
            draw.text((x_s, y_s), c, font=new_font, fill=(255, 255, 255, 255))
            _, _t, _, _b = new_font.getbbox(c)
            y_s += _b

    rotated_layer = layer.rotate(angle, expand=1, center=(rect[0][0], rect[0][1]))

    x_offset = int((img.width - rotated_layer.width) / 2)
    y_offset = int((img.height - rotated_layer.height) / 2)
    img.paste(rotated_layer, (x_offset, y_offset), rotated_layer)
    img = np.expand_dims(np.array(img.convert('1')), axis=2).astype(np.float64)

    return img
