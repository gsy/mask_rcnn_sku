# -*- coding: utf-8 -*-

import json
import os
import shutil
import xml.etree.ElementTree as ET

import xmltodict

images_dir = "/Users/guang/Downloads/store-1-img/fridge_train_data/image_10_14"
xml_dir = "/Users/guang/code/deeplearning/sku/dataset/annotations_10_14"
dataset_dir = "/Users/guang/code/deeplearning/sku/dataset"


def xml_to_json(xml_file):
    f = open(xml_file)
    return xmltodict.parse(f.read())


def find_sku():
    annotations = []
    for rfile in next(os.walk(xml_dir))[2]:
        if not rfile.endswith(".xml"):
            continue

        xml_file = os.path.join(xml_dir, rfile)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        image_file = rfile.replace("xml", "jpg")
        image = os.path.join(images_dir, image_file)
        if os.path.exists(image):
            contain_sku = False
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name == "7fb76eec-83e6-11e8-881b-34363bd1db02":
                    contain_sku = True
                    break

            if contain_sku:
                shutil.copy(image, dataset_dir)
                annotation = xml_to_json(xml_file)
                # 替换成本地文件名
                annotation['annotation']['filename'] = image_file
                annotations.append(annotation)

    json_file = os.path.join(dataset_dir, "annotations.json")
    print(json_file)
    with open(json_file, "w") as wfile:
        json.dump(annotations, wfile)


def rewrite_json():
    dataset_dir = '/home/guang/code/deeplearning/Mask_RCNN/samples/sku/dataset/train'
    json_file = os.path.join(dataset_dir, "annotations.json")
    rfile = open(json_file, "r")
    origin_annotations = json.load(rfile.read())

    annotations = []
    for item in origin_annotations:
        a = item['annotation']
        image = os.path.join(images_dir, a['filename'])
        if os.path.exists(image):
            annotations.append(item)

    outfile = os.path.join(dataset_dir, "real_annotations.json")
    with open(outfile, "w") as wfile:
        json.dump(annotations, wfile)


if __name__ == '__main__':
    # find_sku()
    rewrite_json()
