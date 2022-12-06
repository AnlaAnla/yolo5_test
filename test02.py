import os
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import Profile, check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from werkzeug.utils import secure_filename
from utils.plots import Annotator, colors
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import time
import json
import numpy as np
import easyocr
from datetime import timedelta
import cv2


device = ''
device = select_device(device)

model = DetectMultiBackend('weights/ball_card02.pt', device=device)
img_path = r"D:\Code\ML\images\Mywork\ball_imgs\8256eefc6f6601fb96781fedc5d37f2.jpg"

imgsz = (640, 640)
bs = 1  # batch_size
conf_thres = 0.25
iou_thres = 0.45
max_det = 1000
classes = None
agnostic_nms = True

stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size

dataset = LoadImages(img_path, img_size=imgsz, stride=stride, auto=pt)

# Run inference
model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
seen, windows, dt = 0, [], (Profile(), Profile(), Profile())


for path, im, im0s, vid_cap, s in dataset:
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        pred = model(im)
    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    det = pred[0]
    annotator = Annotator(im0s, line_width=3, example=str(names))
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()

        # Print results
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        # Write results
        *xyxy, conf, cls = det.tolist()[0]
        name = names[int(c)]
        conf = f'{float(conf):.2f}'

        label = name + " " + conf
        annotator.box_label(xyxy, label, color=colors(c, True))

    im0 = annotator.result()

cv2.imwrite('test.jpg', im0)
