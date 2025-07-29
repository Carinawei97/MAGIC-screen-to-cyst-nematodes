import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import shutil

# 1.makeTXT生成四个txt,记录切分情况（只记录图片名字，不记录路径），test、train 、val、trainval
# 2.创建label文件夹，再创建test、train 、val的txt文件，用于保存各部分图片的路径
# 3.将所有图片的xml文件从一个文件夹读取，转存到另一个文件夹（label）变成多个txt文件(yolov5格式的，每个图片一个txt，每一行标注有1个目标的类别及位置)
# 4.将各个部分的图片按照刚才保存的路径，拷贝到目标路径

sets = ['train', 'val']
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
#            "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ["grain"]


# 转换box位置标签
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


# 从xml标签转换到txt标签
def convert_annotation(image_id):
    in_file = open('xmllabel/%s.xml' % (image_id))  # 打开对应的xml文件
    out_file = open('labels/%s.txt' % (image_id), 'w')  # 输出文件,yolov5格式的label
    tree = ET.parse(in_file)
    root = tree.getroot()
    print("image_id", image_id)
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    print("size", size)
    print("w", w)
    print("h", h)

    for obj in root.iter('object'):
        print("")
        print("hhhhhhhhhh")

        if obj.find('difficult'):
            difficult = obj.find('difficult').text
        else:
            difficult = 0

        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)

        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        print("bbbbbb", str(cls_id), bb)

        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')  # 将数据写入输出文件


for year, image_set in sets:
    # 2.创建label文件夹，再创建test、train 、val的txt文件，用于保存各部分图片的路径
    if not os.path.exists('labels/'):  # 创建标签文件夹
        os.makedirs('labels/')

    # 读取TXT文件，保存图片ID
    image_ids = open('grain_datasets/ImageSets/%s.txt' % (image_set)).read().strip().split()  # 读取已经分好的标签
    list_file = open('%s.txt' % (image_set), 'w')  # 保存各个部分数据（test、val、train）图片的路径

    print("image_set", image_set)
    for image_id in image_ids:
        list_file.write('image/%s.jpg\n' % image_id)  # 将每个图片的路径暂时保存到txt------成功
        # 3.将所有图片的xml文件从一个文件夹读取，转存到另一个文件夹（label）变成多个txt文件(yolov5格式的，每个图片一个txt，每一行标注有1个目标的类别及位置)
        convert_annotation(image_id)  # 将xml文件的标签转存到label下的每个TXT
    list_file.close()


# 合并多个文本文件
def mergeTxt(file_list, outfile):
    with open(outfile, 'w') as wfd:
        for f in file_list:
            with open(f, 'r') as fd:
                shutil.copyfileobj(fd, wfd)


# 创建VOC文件夹及子文件夹
# wd = os.getcwd()
wd = 'D:/shanghai_grain/'
data_base_dir = os.path.join(wd, "grain_datasets/")

if not os.path.isdir(data_base_dir):
    os.mkdir(data_base_dir)
img_dir = os.path.join(data_base_dir, "images/")
if not os.path.isdir(img_dir):
    os.mkdir(img_dir)
img_train_dir = os.path.join(img_dir, "train/")
if not os.path.isdir(img_train_dir):
    os.mkdir(img_train_dir)
img_val_dir = os.path.join(img_dir, "val/")
if not os.path.isdir(img_val_dir):
    os.mkdir(img_val_dir)
label_dir = os.path.join(data_base_dir, "labels/")
if not os.path.isdir(label_dir):
    os.mkdir(label_dir)

label_train_dir = os.path.join(label_dir, "train/")
if not os.path.isdir(label_train_dir):
    os.mkdir(label_train_dir)

label_val_dir = os.path.join(label_dir, "val/")
if not os.path.isdir(label_val_dir):
    os.mkdir(label_val_dir)

# 使用train.txt中的图片作为yolov5的训练集
print(os.path.exists('train.txt'))
f = open('train.txt', 'r')
lines = f.readlines()

# 4.将各个部分的图片按照刚才保存的路径，拷贝到目标路径
for line in lines:
    line = line.replace('\n', '')
    if (os.path.exists(line)):
        shutil.copy(line, "grain_datasets/images/train")  # 复制图片到voc_mlw/images/train
        print('coping train img file %s' % line + '\n')

    print("line", line)
    line = line.replace('image', 'labels')  # 复制label，将图片路径名字换成对应TXT的路径名字
    line = line.replace('jpg', 'txt')
    print("line", line)
    if (os.path.exists(line)):
        shutil.copy(line, "grain_datasets/labels/train")
        print('copying train label file %s' % line + '\n')

# 使用test.txt中的图片作为yolov5验证集
print(os.path.exists('test.txt'))
f = open('test.txt', 'r')
lines = f.readlines()

for line in lines:
    line = line.replace('\n', '')
    if (os.path.exists(line)):
        shutil.copy(line, "grain_datasets/images/val")  # line是图片路径，复制图片到 voc_mlw/images/val
        print('coping val img file %s' % line + '\n')

    line = line.replace('image', 'labels')  # 复制label，将图片路径名字换成对应TXT的路径名字
    line = line.replace('jpg', 'txt')
    if (os.path.exists(line)):
        shutil.copy(line, "grain_datasets/labels/val")
        print('copying val img label  %s' % line + '\n')