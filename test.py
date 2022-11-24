import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/ball_card.pt')

im1 = Image.open(r"D:\Pictures\f9ae1d3d-914d-4069-839a-b3ed3a5c7323.jpg")  # PIL image

# Inference
results = model([im1], size=640) # batch of imag

print(results)
