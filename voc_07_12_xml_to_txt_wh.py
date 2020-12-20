import xml.etree.ElementTree as ET
import os

sets = [('2019', 'train', 'train_list', 'train_l'), ('2019', 'val', 'val_list', 'val_l'),
        ('2019', 'test', 'test_list', 'test_l')]

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def convert_annotation(year, image_id, list_file, jpg_label):
    in_file = open('my_imgs&labels/%s/labels/%s/%s.xml' % (year, jpg_label, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    xmlsize = root.find('size')
    d = (int(xmlsize.find('width').text), int(xmlsize.find('height').text))
    list_file.write(" " + " ".join([str(c) for c in d]))
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        list_file.write(" " + str(cls_id) + " " + " ".join([str(a) for a in b]))


wd = os.getcwd()

for year, image_set, name_list, jpg_label in sets:
    data_base_dir = ("my_imgs&labels/%s/imgs/%s" % (year, image_set))
    file_list = []
    write_file_name = ('my_imgs&labels/%s/imgs/%s/%s.txt' % (year, image_set, name_list))
    write_file = open(write_file_name, "w")
    for file in os.listdir(data_base_dir):
        if file.endswith(".jpg"):
            index = file.rfind('.')
            file = file[:index]
            file_list.append(file)
    number_of_lines = len(file_list)
    for current_line in range(number_of_lines):
        write_file.write(file_list[current_line] + '\n')
    write_file.close()
    image_ids = open('my_imgs&labels/%s/imgs/%s/%s.txt' % (year, image_set, name_list)).read().strip().split()
    list_file = open('my_imgs&labels/%s/final_datas_wh/%s.txt' % (year, image_set), 'w')
    line_ind = 0
    for image_id in image_ids:
        list_file.write('%d %s/my_imgs&labels/%s/imgs/%s/%s.jpg' % (line_ind, wd, year, image_set, image_id))
        convert_annotation(year, image_id, list_file, jpg_label)
        list_file.write('\n')
        line_ind += 1
    list_file.close()

txt_path_train = './my_imgs&labels/2019/final_datas_wh/train.txt'
txt_path_val = './my_imgs&labels/2019/final_datas_wh/val.txt'
txt_path_test = './my_imgs&labels/2019/final_datas_wh/test.txt'

# next codelines is for 'unclear boxes' postprecessing.
# the lines shown is where you should delete, it's my suggestion.
with open(txt_path_train, 'r') as fileread:
    while True:
        line = fileread.readline()
        if not line:
            break
        cur_line_num = line.strip().split(' ')
        if len(cur_line_num) < 5:
            print(cur_line_num[0])
with open(txt_path_val, 'r') as fileread:
    while True:
        line = fileread.readline()
        if not line:
            break
        cur_line_num = line.strip().split(' ')
        if len(cur_line_num) < 5:
            print(cur_line_num[0])
with open(txt_path_test, 'r') as fileread:
    while True:
        line = fileread.readline()
        if not line:
            break
        cur_line_num = line.strip().split(' ')
        if len(cur_line_num) < 5:
            print(cur_line_num[0])

        # file path like:
        # my_imgs%labels->2019->{final_datas_wh, imgs, labels}->final_datas_wh->{train.txt, val.txt, test.txt}
        #                                                     ->imgs->{train, val, test}
        #                                                     ->labels->{train_l, val_l, test_l}
        # final_datas_wh is what you need in the end.
        # imgs is where you put your images in, and labels is the same thing.
        # note that you must be sure of making the same name of every image and label.
        # finally, you got the txt files, enjoy your life!!!
        # more info: https://blog.csdn.net/qq_43322615/article/details/94567969
