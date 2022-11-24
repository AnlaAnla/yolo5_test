import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def image_add_text(img, results, text_color=(255, 125, 0), text_size=30):
    name = results.pandas().xyxy[0].to_numpy()[0][-1]
    rect = results.pandas().xyxy[0].to_numpy()[0][:4]

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("util/SmileySans-Oblique.ttf", text_size)
    draw.rectangle(list(rect), outline=(0, 255, 0), width=3)
    draw.text((rect[0], rect[1]), name, text_color, font=font)

    return img

model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/ball_card02.pt')
im1 = Image.open(r"D:\Pictures\f9ae1d3d-914d-4069-839a-b3ed3a5c7323.jpg")  # PIL image

# Inference
results = model([im1], size=640) # batch of imag
#
print(results)
img = image_add_text(im1, results)
img.show()

