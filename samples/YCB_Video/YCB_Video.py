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
import math
import numpy as np
import cv2
import imgaug
import skimage.io
import skimage.color
import keras.layers as KL
import depth_aware_operations.da_convolution as da_conv
import depth_aware_operations.da_avg_pooling as da_avg_pool

DCKL = da_conv.keras_layers
DPKL = da_avg_pool.keras_layers

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
#                                                      Backbone                                                        #
########################################################################################################################

def da_resnet_graph(self, input_image, stage5=False, train_bn=True, depth_image=None):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    print(self)
    print(input_image)
    assert depth_image is not None
    architecture = "resnet101"
    # Stage 1
    x = DCKL.DAConv2D(32, (12, 12), strides=(1, 1), padding="same", name='conv1a', depth_image=depth_image)(input_image)
    x = KL.ZeroPadding2D((3, 3))(x)
    d = KL.ZeroPadding2D((3, 3))(depth_image)
    x = DCKL.DAConv2D(64, (7, 7), strides=(2, 2), name='conv1b', use_bias=True, depth_image=d)(x)
    x = modellib.BatchNorm(name='bn_conv1')(x, training=train_bn)
    d = KL.AveragePooling2D((7, 7), strides=(2, 2))(d)
    x = KL.Activation('relu')(x)
    # TODO: use DAAvgPooling here
    C1 = x = DPKL.DAAveragePooling2D(d, pool_size=(3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = modellib.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = modellib.identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = modellib.identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = modellib.conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = modellib.identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = modellib.identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = modellib.identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = modellib.conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = modellib.identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
        # x = KL.Lambda(lambda y: tf.Print(y, [tf.shape(y)], message="This is the shape of x: "))(x)
    C4 = x
    # Stage 5
    if stage5:
        x = modellib.conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = modellib.identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = modellib.identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


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
    IMAGES_PER_GPU = 2
    # do not resize the images, as they are all the same size
    # TODO: Apparently, something goes pretty wrong when IMAGE_RESIZE_MODE is none; mrcnn_bbox_loss and mrcnn_mask_loss
    # are always zero then
    IMAGE_RESIZE_MODE = "square"#"none"
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640
    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 21  # 21 Objects were selected from the original YCB Dataset

    #TRAIN_ROIS_PER_IMAGE = 100
    USE_DEPTH_AWARE_OPS = True

    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    # IMAGE_CHANNEL_COUNT = 4
    #BACKBONE = da_resnet_graph
    #COMPUTE_BACKBONE_SHAPE = lambda config, image_shape : np.array(
    #    [[int(math.ceil(image_shape[0] / stride)),
    #      int(math.ceil(image_shape[1] / stride))]
    #     for stride in config.BACKBONE_STRIDES])



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
    default_weights = "last"
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
        # Download weights file
        if not os.path.exists(model_path):
            utils.download_trained_weights(model_path)
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
    if args.model.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(model_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
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
        layers = "1-2"
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE/5.,
                    epochs=230,
                    layers=layers,
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        #if args.depth_aware:
        #    layers = "4+"
        #else:
        #    layers = '4+'
        #print("Fine tune Resnet stage 4 and up")
        #model.train(dataset_train, dataset_val,
        #            learning_rate=config.LEARNING_RATE,
        #            epochs=100,
        #            layers=layers,
        #            augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=280,
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
