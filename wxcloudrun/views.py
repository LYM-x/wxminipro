from datetime import datetime
from flask import render_template, request
from run import app
from wxcloudrun.response import make_succ_empty_response, make_succ_response, make_err_response
import os
from cv2 import *
from skimage import io
import json
import requests
#import detect.py

@app.route('/')
def index():
    return ("hello")


@app.route('/api/photo', methods=['GET'])
def count():
    # 获取请求体参数
    img_src = request.args.get("url")
    #下载图片
    savesplit=img_src.split('/')[-1]
    r = requests.get(img_src,verify=False)
    outputoath='api/out'+savesplit
    # 写入图片
    with open("images/input.jpg", "wb") as f:
        f.write(r.content)
    os.system("python3 detect.py")
    #读取识别结果
    
    #将当前目录下的一张图片上传到云
    #获取token,上传图片
    response = requests.get('https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid=wx0caa0328c1763b7d&secret=7b24bcebbcbdb257f1cf931b55c2d7e7',verify=False)
    data ={
        "env": "prod-1ghgatwy02a74c12",#修改成自己的环境id
        "path": outputoath
    }
    #转json
    data = json.dumps(data) #一定要把参数转为json格式，不然会请求失败
    response = requests.post("https://api.weixin.qq.com/tcb/uploadfile?access_token="+response.json()['access_token'],data,verify=False)

    data2={
        "Content-Type":(None,".jpg"), #此处为上传文件类型
        "key": (None,outputoath),
        "Signature": (None,response.json()['authorization']),
        'x-cos-security-token': (None,response.json()['token']),
        'x-cos-meta-fileid': (None,response.json()['cos_file_id']),
        'file': (outputoath,open(r"runs/detect/exp/output.jpg","rb"))
    }
    response2 = requests.post(response.json()['url'], files=data2,verify=False) #此处files提交的为表单数据，不为json数据，json数据或其他数据会报错
    #注意保存
    fileid = response.json()["file_id"]
    #可在小程序端调用获取图片
    bpercent=-1;mpercent=-1
    ff=open("result.txt", "r")
    for line in ff:
        line=line.strip('\n')   #将\n去掉
        words=(line.split(' '))   #分割
        if words[0]=='b':bpercent=float(words[1])
        elif words[0]=='m':mpercent=float(words[1])
    ff.close()
    with open("result.txt", "r") as ff:
        status=ff.read()
    ff.close()
    #读取长宽文件
    with open("runs/detect/exp/detect_location/changkuan.txt", "r") as FF:
        number=FF.read()
    numbers=number.split(' ')   #拆分
    wid=float(numbers[0]);len=float(numbers[1])
    space=wid*len
    data = {
        "geturl":img_src,
        "fileid": fileid,
        "bpercent":bpercent,    #良性b 恶性m
        "mpercent":mpercent,
        "len":len,
        "wid":wid,
        "space":space
    }
    res_json = json.dumps(data)
    return res_json
