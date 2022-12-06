import os
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from models.common import DetectMultiBackend
from utils.augmentations import classify_transforms
from utils.dataloaders import LoadImages
from utils.general import Profile
from werkzeug.utils import secure_filename

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
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
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

    cls_text = ""
    for i in top3pre:
        cls_text += ("{}: {:.3}%".format(classes_names[i], prob_list[i] * 100)) + "     "
    return cls_text


# 给图片添加文本和方框
def image_add_text(img, name, rect, text_color=(255, 125, 0), text_size=30):


    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("utils/SmileySans-Oblique.ttf", text_size)
    draw.rectangle(list(rect), outline=(0, 255, 0), width=3)
    draw.text((rect[0], rect[1]), name, text_color, font=font)

    return img


def dtect_img(model, img, size=640):
    results = model([img], size=size)
    print(results.pandas().xyxy)
    # 返回预测结果
    try:
        name = results.pandas().xyxy[0].to_numpy()[0][-1]
        rect = results.pandas().xyxy[0].to_numpy()[0][:4]
        # 标注后的结果图像
        pred_img = image_add_text(img, name, rect)

        img_name = str(get_num()) + '.jpg'
        pred_img.save(os.path.join('./static/images', img_name))
        predict_txt = name
    except:
        predict_txt = "无"
        img_name = ''

    return (img_name, predict_txt)




@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':

        # 文件上传及保存
        f = request.files['file']
        user_input = request.form.get("name")

        upload_path = save_requestImg(f)


        # 访问次数记录
        set_num(get_num() + 1)

        # 使用Opencv转换一下图片格式和名称
        img = Image.open(upload_path)

        # 是否执行OCR
        if use_ocr:
            read_text = reader.readtext(np.array(img), detail=0,
                                        allowlist=allow_list,rotation_info=[-30, 30])
        else:
            read_text = '没有设置文字读取'


        # 模型读取图像
        img_name, predict_txt = dtect_img(model_detect, img=img)

        if predict_txt == 'PRIZM':
            cls_text = predict_oneImg(model_cls_p, img_path=upload_path)
        elif predict_txt == 'MOSAIC':
            cls_text = predict_oneImg(model_cls_m, img_path=upload_path)
        else:
            cls_text = ''
        print(img_name,"分类： ", cls_text)


        text = user_input + " | " + predict_txt
        print(predict_txt)
        return render_template('upload_ok.html', userinput=text, cls_text=cls_text,
                               ort_text = str(read_text), img_name=img_name, val1=time.time())


    return render_template('upload.html')


if __name__ == '__main__':
    model_detect = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/ball_card02.pt')
    model_cls_p = DetectMultiBackend('weights/ball_card_cls2prime.pt')
    model_cls_m = DetectMultiBackend('weights/ball_card_cls2mosaic.pt')

    reader = easyocr.Reader(['en'])
    allow_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U',
                  'V', 'W', 'X', 'Y', 'Z',
                  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                  'u',
                  'v', 'w', 'x', 'y', 'z',
                  "'", "\"", ',', '?', '.', ' ']



    # 是否使用OCR，使用后速度会下降很多，相对于yolo
    use_ocr = False
    app.debug = False
    print(torch.cuda.is_available())
    app.run()