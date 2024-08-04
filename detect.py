import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    #save_dir = str('run/detect/exp')
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / '' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / '' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string

            ####################### 自己改的部分
            location_center_dir = str(save_dir) + '/detect_location'
            if not os.path.exists(location_center_dir):
                os.makedirs(location_center_dir)
            location_center_path = location_center_dir + '/' + 'changkuan' + (
                '' if dataset.mode == 'image' else f'_{frame}')  #
            flocation = open(location_center_path + '.txt', 'a')
            #######################

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    
                    ######## 自己改的部分
                    x0 = (int(xyxy[0].item()) + int(xyxy[2].item())) / 2
                    y0 = (int(xyxy[1].item()) + int(xyxy[3].item())) / 2  # 中心点坐标(x0, y0)

                    chang = int(xyxy[2].item()) - int(xyxy[0].item())
                    kuan = int(xyxy[3].item()) - int(xyxy[1].item())
                    # class_index = cls  # 获取属性
                    # object_name = names[int(cls)]  # 获取标签名如：person
                    label = int(cls)  # 对应每个物体的标签对应的数字label，如0
                    x0 = format(x0/img.shape[1], '.6f')  #此处我要保存和训练标注的txt一样的格式，故保留六位小数
                    y0 = format(y0/img.shape[0], '.6f')
                    chang = format(chang/img.shape[1], '.6f') # img.shape[1]和img.shape[0]为我的图片长和宽
                    kuan = format(kuan/img.shape[0], '.6f')
                    if label==0:  # 在这里我只需要保存person的信息，可以删去，也可以自己更改
                        # flocation.write(str(label) + ' ' + str(x0) + ' ' + str(y0) + ' '+str(chang)+' ' +str(kuan)+'\n')
                        flocation.write(str(chang)+' ' +str(kuan))
                    ##########
                    
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('*.txt')))} labels saved to {save_dir / ''}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")
        print(f"Results saved to {save_dir}")

    print(f'Done. ({time.time() - t0:.3f}s)')

######################################裁剪#######################################################################
import os
def cut():
    img_path = 'images'  # 图片路径
    label_path = 'runs/detect/exp'  # txt文件路径
    save_path = 'cut'    # 保存路径

    img_total = []
    label_total = []
    imgfile = os.listdir(img_path)
    labelfile = os.listdir(label_path)

    for filename in imgfile:
        name, type = os.path.splitext(filename)
        if type == ('.jpg' or '.png'):
            img_total.append(name)
    for filename in labelfile:
        name, type = os.path.splitext(filename)
        if type == '.txt':
            label_total.append(name)



    for _img in img_total:
        if _img in label_total:
            filename_img = _img + '.jpg'
            path = os.path.join(img_path, filename_img)
            img = cv2.imread(path)  # 读取图片，结果为三维数组
            filename_label = _img + '.txt'
            w = img.shape[1]  # 图片宽度(像素)
            h = img.shape[0]  # 图片高度(像素)
            n = 1
            # 打开文件，编码格式'utf-8','r+'读写
            with open(os.path.join(label_path, filename_label), "r+", encoding='utf-8', errors="ignor") as f:
                for line in f:
                    msg = line.split(" ")  # 根据空格切割字符串，最后得到的是一个list
                    x1 = int((float(msg[1]) - float(msg[3]) / 2) * w)  # x_center - width/2
                    y1 = int((float(msg[2]) - float(msg[4]) / 2) * h)  # y_center - height/2
                    x2 = int((float(msg[1]) + float(msg[3]) / 2) * w)  # x_center + width/2
                    y2 = int((float(msg[2]) + float(msg[4]) / 2) * h)  # y_center + height/2
                    filename_last = _img + "_" + str(n) + ".jpg"
                    print(filename_last)
                    img_roi = img[y1:y2, x1:x2] # 剪裁，roi:region of interest
                    cv2.imwrite(os.path.join(save_path, filename_last), img_roi)
                    n = n + 1
        else:
            continue

########################################convnext分类#####################################################
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import convnext_tiny as create_model


def classification(img_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    num_classes = 2
    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    
    #img_path = input('Input image filename:')
    #img_path = 'cut/1_1.jpg'
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
   # [N, C, H, W]
    img = data_transform(img)
     # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

     # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

     # create model
    model = create_model(num_classes=num_classes).to(device)
     # load model weights
    model_weight_path = "weights/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    f = open("result.txt","w")
    for i in range(len(predict)):
        print("{:10} {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
        f.write("{:2}{:.3}\n".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    f.close()
    #plt.savefig('save'+'/'+filename)
    #plt.show()
    #return class_indict[str(predict_cla)]



if __name__ == '__main__':

    import shutil  
    if os.path.exists(r'runs/detect/exp'):
      shutil.rmtree('runs/detect/exp')  
    shutil.rmtree('cut')
    os.mkdir('cut')

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.70, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt',default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
    
    #重命名为output.jpg
    yolo_path = 'runs/detect/exp'
    files  = os.listdir(yolo_path)
    for filename in files:
       name, type = os.path.splitext(filename)
       if type == ('.jpg' or '.png'):
             os.rename(os.path.join(yolo_path,filename),os.path.join(yolo_path,"output")+".jpg")

################裁剪##################
    cut()
################分类####################
    #classification()
    for filename in os.listdir('cut'):
        classification('cut'+'/'+filename)

######################清空文件夹下的文件###############
   # shutil.rmtree('images')  
   # os.mkdir('images')

