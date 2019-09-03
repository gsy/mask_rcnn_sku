# -*- coding: utf-8 -*-

import json
import os

dataset_dir = "/Users/guang/code/deeplearning/sku/dataset"


annotations = json.load(open(os.path.join(dataset_dir, "annotations.json")))
for a in annotations:
    # image path
    print(a)
    break
