from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
# from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
import os
import torch
from PIL import Image, ImageDraw, ImageFont
import time
import json
import numpy as np
import easyocr
from datetime import timedelta

# 设置允许的文件格式
ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'PNG', 'bmp'}



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
# run_with_ngrok(app)
# 设置静态文件缓存过期时间
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

# 给图片添加文本和方框
def image_add_text(img, name, rect, text_color=(255, 125, 0), text_size=30):


    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("util/SmileySans-Oblique.ttf", text_size)
    draw.rectangle(list(rect), outline=(0, 255, 0), width=3)
    draw.text((rect[0], rect[1]), name, text_color, font=font)

    return img



# @app.route('/upload', methods=['POST', 'GET'])
@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        # upload_path = os.path.join(basepath, 'static/images','test.jpg')  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)


        # 访问次数记录
        img_num = get_num()
        img_num += 1
        set_num(img_num)

        # 使用Opencv转换一下图片格式和名称
        img = Image.open(upload_path)

        if use_ocr:
            read_text = reader.readtext(np.array(img), detail=0)
        else:
            read_text = '没有设置文字读取'

        # 模型读取图像
        results = model([img], size=640)
        print(results.pandas().xyxy)
        # 返回预测结果
        try:
            name = results.pandas().xyxy[0].to_numpy()[0][-1]
            rect = results.pandas().xyxy[0].to_numpy()[0][:4]
            # 标注后的结果图像
            pred_img = image_add_text(img, name, rect)

            img_name = str(img_num) + '.jpg'
            pred_img.save(os.path.join('./static/images', img_name))
            predict_txt = name
        except:
            predict_txt = "无"
            img_name = ''


        text = user_input + " | " + predict_txt
        print(predict_txt)
        return render_template('upload_ok.html', userinput=text, ort_text = str(read_text), img_name=img_name, val1=time.time())


    return render_template('upload.html')


if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/ball_card02.pt')
    reader = easyocr.Reader(['en'])

    # 是否使用OCR，使用后速度会下降很多，相对于yolo
    use_ocr = True
    # app.debug = True

    app.run()