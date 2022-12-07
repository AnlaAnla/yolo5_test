import os
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from models.common import DetectMultiBackend
from utils.augmentations import classify_transforms
from utils.dataloaders import LoadImages
from utils.general import Profile, check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from werkzeug.utils import secure_filename
from utils.plots import Annotator, colors
import cv2

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import time
import json
import numpy as np
import easyocr
from datetime import timedelta

'''
    WARNING
    下面三行用在window系统，在linux有问题的话，删掉
'''
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath
# -----------------------------------------

# 设置允许的文件格式
ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'PNG', 'bmp'}



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)


# flask全局变量太难用，用文件保存数字
def set_num(img_num):
    record_dict = {"img_num": img_num}
    with open("static/record.json", "w") as f:
        json.dump(record_dict, f)
        print("加载入文件完成...")

def get_num():
    with open("static/record.json", 'r') as load_f:
        load_dict = json.load(load_f)
        img_num = load_dict['img_num']
    return img_num

# 保存上传图片并返回保存路径，默认路径为 'static/images'
def save_requestImg(f):
    if not (f and allowed_file(f.filename)):
        return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

    user_input = request.form.get("name")

    basepath = os.path.dirname(__file__)  # 当前文件所在路径

    upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
    # upload_path = os.path.join(basepath, 'static/images','test.jpg')  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
    f.save(upload_path)
    return upload_path

# prizm 分类
def predict_oneImg(model, img_path, imgsz = (224, 224), dt = (Profile(), Profile(), Profile())):
    dataset = LoadImages(img_path, img_size=224,
                         transforms=classify_transforms(imgsz[0]))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.Tensor(im).to(model.device)
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            results = model(im)

        # Post-process
        with dt[2]:
            pred = F.softmax(results, dim=1)

    classes_names = model.names
    prob_list = pred.tolist()[0]
    top3pre = pred.argsort(descending=True).tolist()[0][:3]

    cls_text = []
    for i in top3pre:
        cls_text.append("{}: {:.3}%".format(classes_names[i], prob_list[i] * 100))
    return cls_text


def dtect_img(model, img_path, size=640):
    img_name = ''
    name = ''
    conf = 0


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

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # 数据读取
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
            print(xyxy, "可信度：", conf, name)

            # 图像标注
            label = name + " " + conf
            annotator.box_label(xyxy, label, color=colors(c, True))


            img_name = str(get_num()) + '.jpg'
            save_path = (os.path.join('./static/images', img_name))

            # 保存图片
            img = annotator.result()
            cv2.imwrite(save_path, img)


    return img_name, name, conf




@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':

        # 文件上传及保存
        f = request.files['file']
        user_input = request.form.get("name")

        upload_path = save_requestImg(f)

        # 访问次数记录
        set_num(get_num() + 1)


        # 是否执行OCR
        if use_ocr:
            # 使用Opencv转换一下图片格式和名称
            img = Image.open(upload_path)
            read_text = reader.readtext(np.array(img), detail=0,
                                        allowlist=allow_list,rotation_info=[-30, 30])
        else:
            read_text = '没有设置文字读取'


        # 模型读取图像
        img_name, predict_name, predict_conf = dtect_img(model_detect, img_path=upload_path)

        if predict_name == 'PRIZM':
            cls_text = predict_oneImg(model_cls_p, img_path=upload_path)
        elif predict_name == 'MOSAIC':
            cls_text = predict_oneImg(model_cls_m, img_path=upload_path)
        else:
            cls_text = ''
        print(img_name,"分类： ", cls_text)


        text = user_input + " | " + predict_name
        print(text)
        return render_template('upload_ok.html', userinput=text, cls_text=cls_text,
                               ort_text = str(read_text), img_name=img_name, val1=time.time())


    return render_template('upload.html')

@app.route('/image_api', methods=['POST', 'GET'])  # 添加路由
def imahe_api():
    if request.method == 'POST':

        # 文件上传及保存
        f = request.files['file']
        upload_path = save_requestImg(f)

        # 访问次数记录
        set_num(get_num() + 1)


        # 是否执行OCR
        if use_ocr:
            # 使用Opencv转换一下图片格式和名称
            img = Image.open(upload_path)
            read_text = reader.readtext(np.array(img), detail=0,
                                        allowlist=allow_list,rotation_info=[-30, 30])
        else:
            read_text = '没有设置文字读取'


        # 模型读取图像
        img_name, predict_name, predict_conf = dtect_img(model_detect, img_path=upload_path)

        if predict_name == 'PRIZM':
            cls_text = predict_oneImg(model_cls_p, img_path=upload_path)
        elif predict_name == 'MOSAIC':
            cls_text = predict_oneImg(model_cls_m, img_path=upload_path)
        else:
            cls_text = ''
        print(img_name,"分类： ", cls_text)


        result = {  "serise": predict_name,
                    "category": cls_text,
                    "content": read_text,
                    "imageurl": "static/images/" + img_name }

        return json.dumps(result)


    return render_template('img_api.html')



if __name__ == '__main__':

    device = ''
    device = select_device(device)


    # model_detect = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/ball_card02.pt')
    model_detect = DetectMultiBackend('weights/ball_card02.pt', device=device)
    model_cls_p = DetectMultiBackend('weights/ball_card_cls2prime.pt', device=device)
    model_cls_m = DetectMultiBackend('weights/ball_card_cls2mosaic.pt', device=device)

    reader = easyocr.Reader(['en'])
    allow_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U',
                  'V', 'W', 'X', 'Y', 'Z',
                  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                  'u',
                  'v', 'w', 'x', 'y', 'z',
                  "'", "\"", ',', '?', '.', ' ']



    # 是否使用OCR，使用后速度会下降很多，相对于yolo
    use_ocr = True
    app.debug = True
    print(torch.cuda.is_available())
    app.run()
