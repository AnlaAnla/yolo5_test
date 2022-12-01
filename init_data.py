import os
import json

def set_num(img_num):
    record_dict = {"img_num": img_num}
    with open("static/record.json", "w") as f:
        json.dump(record_dict, f)
        print("加载入文件完成...")

# 清空图片文件，并且将record设置为零
if __name__ == '__main__':
    path = 'static/images'
    img_list = os.listdir(path)
    img_list.remove('index.txt')

    for name in img_list:
        os.remove(os.path.join(path, name))

    set_num(0)