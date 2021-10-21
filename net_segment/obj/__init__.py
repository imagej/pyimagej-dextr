import imagej
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
    
    
    return

def open_image(path: str, resize_width=256, resize_height=256, show=False) -> Image.Image:
    """
    Accepts only 8-bit images
    """
    img = imagej.ij.io().open(path)
    img_array = imagej.ij.py.from_java(img)
    img_array = img_array.data
    pil_image = Image.fromarray(img_array)
    pil_image = ImageOps.fit(pil_image, (resize_width, resize_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert("RGB")

    if show:
        show_image(pil_image)

    return pil_image_rgb