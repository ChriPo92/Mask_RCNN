"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import numpy as np
import cv2
import imgaug
import skimage.io
import skimage.color

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

########################################################################################################################
#                                                       Utility                                                        #
########################################################################################################################

def load_id_classes_dict(path):
    id_class = {}
    with open(path) as f:
        for i, line in enumerate(f):
            ycb_id = int(line[:3])
            name = line[4:-1]
            id_class[i] = {"ycb_id": ycb_id, "name": name}
    return id_class

def load_image_ids(path):
    l = []
    with open(path) as f:
        for line in f:
            l.append(line[:-1])
    return l




########################################################################################################################
#                                                   Configurations                                                     #
########################################################################################################################


class YCBVConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "YCBV"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    # do not resize the images, as they are all the same size
    IMAGE_RESIZE_MODE = "none"
    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 21  # 21 Objects were selected from the original YCB Dataset

    TRAIN_ROIS_PER_IMAGE = 130
    USE_DEPTH_AWARE_OPS = True
    # IMAGE_CHANNEL_COUNT = 4

    # Image mean (RGB)



########################################################################################################################
#                                                        Dataset                                                       #
########################################################################################################################

class YCBVDataset(utils.Dataset):
    def load_ycbv(self, dataset_dir, subset, class_ids=None, use_synthetic=False, use_rgbd=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        class_ids: If provided, only loads images that have the given classes.
        use_synthetic: If True uses the synthetic data in the data_syn folder (exclusively?)
        """
        self.use_rgbd=use_rgbd
        image_sets = os.path.join(dataset_dir, "image_sets/")
        image_dir = os.path.join(dataset_dir, "data/")
        classes_dict = load_id_classes_dict(os.path.join(image_sets, "classes.txt"))
        # Load all classes or a subset?
        if class_ids is None:
            # All classes
            class_ids = sorted(classes_dict.keys())

        # Add classes
        for i in class_ids:
            self.add_class("YCBV", int(i), classes_dict[i]["name"])

        # load image ids
        # the training images are listed in "train", all images in "trainval", the images used for validation in
        # "minival", the evaluation/test set in "keyframe" and "val" = "minival" + "keyframe"
        assert subset in ["train", "trainval", "val", "minival", "keyframe"]
        img_id_path = os.path.join(image_sets, "%s.txt"%subset)
        image_ids = load_image_ids(img_id_path)

        # Add images
        for i in image_ids:
            self.add_image(
                "YCBV", image_id=i,
                path=os.path.join(image_dir, i + "-color.png"),
                width=640,
                height=480,
                annotations=os.path.join(image_dir, i + "-label.png"),
                depth=os.path.join(image_dir, i + "-depth.png"),
                meta=os.path.join(image_dir, i + "-meta.mat"),
                box=os.path.join(image_dir, i + "-box.txt"))

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a YCB image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "YCBV":
            return super(YCBVDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotation_path = self.image_info[image_id]["annotations"]
        # labels are saved in an image, as a mask of all objects, where each objects
        # mask consists of its corresponding class_id
        ann = cv2.imread(annotation_path, 0)
        classes = np.unique(ann)[1:] # drop the zero (background)
        masks = []
        class_ids = []
        for i in classes:
            masks.append((ann == i).reshape(480, 640, 1))
            class_ids.append(i)
        return np.concatenate(masks, axis=2), np.array(class_ids, np.int32)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        if self.use_rgbd:
            depth = skimage.io.imread(self.image_info[image_id]['depth'])
            image = np.concatenate((image, np.expand_dims(depth / 10000, 2)), axis=2)
        return image

########################################################################################################################
#                                                       Training                                                       #
########################################################################################################################


if __name__ == '__main__':
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    import argparse
    default_weights = "logs/ycbv20181019T1806/mask_rcnn_ycbv_0020.h5"
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on YCB Video Dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on YCBV")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/ycbv/",
                        help='Directory of the YCB Video dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--depth_aware', required=False, type=str2bool,
                       default=False,
                       metavar="<image count>",
                       help='Train with depth-aware operations')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("Depth Awareness: ", args.depth_aware)


    # Configurations
    if args.command == "train":
        config = YCBVConfig()
    else:
        class InferenceConfig(YCBVConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.

        dataset_train = YCBVDataset()
        dataset_train.load_ycbv(args.dataset, "train", use_rgbd=args.depth_aware)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = YCBVDataset()
        val_type = "minival"
        dataset_val.load_ycbv(args.dataset, val_type, use_rgbd=args.depth_aware)
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        # TODO: if image is flipped, the 6d Pose changes --> make amends
        augmentation = imgaug.augmenters.Fliplr(0.5)
        # augmentation = None
        # *** This training schedule is an example. Update to your needs ***
        # Training - Stage 1
        print("Training network heads")
        if args.depth_aware:
            layers = "heads_c"
        else:
            layers = "heads"
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers=layers,
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        if args.depth_aware:
            layers = "4+"
        else:
            layers = '4+'
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=100,
                    layers=layers,
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=180,
                    layers='all',
                    augmentation=augmentation)

    # elif args.command == "evaluate":
    #     # Validation dataset
    #     dataset_val = CocoDataset()
    #     val_type = "val" if args.year in '2017' else "minival"
    #     coco = dataset_val.load_coco(args.dataset, val_type, year=args.year, return_coco=True, auto_download=args.download)
    #     dataset_val.prepare()
    #     print("Running COCO evaluation on {} images.".format(args.limit))
    #     evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
