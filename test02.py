import json


def set_num(img_num):
     record_dict = {"img_num": img_num}
     with open("static/images/record.json","w") as f:
          json.dump(record_dict,f)
          print("加载入文件完成...")

def get_num():
     with open("static/images/record.json",'r') as load_f:
          load_dict = json.load(load_f)
          img_num = load_dict['img_num']
     return img_num