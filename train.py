# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
ROOT_DIR = os.path.abspath("../..") # noqa
sys.path.append(ROOT_DIR)         # noqa

from mrcnn.config import Config
from mrcnn import model as modellib, utils
import numpy as np
import skimage.draw
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "sku/logs")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


############################################################
#  Configurations
############################################################


class SkuConfig(Config):
    # Give the configuration a recognizable name
    NAME = "sku"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + skus

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class SkuDataset(utils.Dataset):
    def load_sku(self, dataset_dir, subset):
        """Load a subset of the sku dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # 只认一种 sku: 百岁山矿泉水570ml, 7fb76eec-83e6-11e8-881b-34363bd1db02
        # 读取有百岁山的图片, 一共566张
        self.add_class("sku", 1, "百岁山矿泉水")

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        # Load annotations
        annotations = json.load(open(os.path.join(dataset_dir, "real_annotations.json")))
        for item in annotations:
            a = item['annotation']
            print("annotation", a)
            # image path
            image_path = os.path.join(dataset_dir, a['filename'])
            bboxes = [item['bndbox'] for item in a['object'] if item['name'] == "7fb76eec-83e6-11e8-881b-34363bd1db02"]
            # add image 是使用一个字典保存 image 的信息，在 load_mask 里面使用了
            self.add_image(
                "sku",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=int(a['size']['width']),
                height=int(a['size']['height']),
                bboxes=bboxes)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a sku dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "sku":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["bboxes"])],
                        dtype=np.uint8)
        for i, bbox in enumerate(info["bboxes"]):
            # Get indexes of pixels inside the polygon and set them to 1
            ys = np.array([int(bbox['ymin']), int(bbox['ymin']), int(bbox['ymax']), int(bbox['ymax'])])
            xs = np.array([int(bbox['xmin']), int(bbox['xmax']), int(bbox['xmax']), int(bbox['xmin'])])

            rr, cc = skimage.draw.polygon(ys, xs)
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "sku":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = SkuDataset()
    dataset_train.load_sku(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SkuDataset()
    dataset_val.load_sku(args.dataset, "val")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train,
                dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads')


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect skus.')
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/train/dataset/",
                        help='Directory of the train dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    config = SkuConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train
    train(model)
