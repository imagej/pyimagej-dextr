import xarray as xr
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

def show_image(image: Image.Image):
    fig = plt.figure(figsize=(10,8))
    plt.grid(False)
    plt.imshow(image)
    plt.show()

    return None

def bounding_box(image: Image.Image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str_list=()):
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size
    (left, right, top, bottom) = (xmin * img_width, xmax * img_width, ymin * img_height, ymax * img_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)
    # correct for boxes drawn outside the image size
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
    
    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # reverse list and print from bottom to top
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)
        draw.text((left + margin, text_bottom - text_height - margin), display_str, fill="black", font=font)
        text_bottom -= text_height - 2 * margin

    return None

def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"), int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box(
                image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str])
        np.copyto(image, np.array(image_pil))

    return None

def open_image(ij_instance, path: str, resize_width=256, resize_height=256, show=False) -> Image.Image:
    """
    Accepts only 8-bit images and resizes them.
    """
    img = ij_instance.ij.io().open(path)
    img_array = ij_instance.ij.py.from_java(img)
    img_array = img_array.data
    pil_image = Image.fromarray(img_array)
    pil_image = ImageOps.fit(pil_image, (resize_width, resize_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert("RGB")

    if show:
        show_image(pil_image)

    return pil_image_rgb

def load_image():

    return None