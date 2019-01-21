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
import time
import math
import numpy as np
import cv2
import pandas as pd
import skimage.io
import skimage.color
import scipy.io as scio
import keras.layers as KL
import keras.backend as KB
import depth_aware_operations.da_convolution as da_conv
import depth_aware_operations.da_avg_pooling as da_avg_pool
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.client import timeline

from tqdm import tqdm

DCKL = da_conv.keras_layers
DPKL = da_avg_pool.keras_layers

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize

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
            # 0 is background
            id_class[i+1] = {"ycb_id": ycb_id, "name": name}
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
    def load_ycbv(self, dataset_dir, subset, class_ids=None, use_annotation="label", use_rgbd=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        class_ids: If provided, only loads images that have the given classes.
        use_synthetic: If True uses the synthetic data in the data_syn folder (exclusively?)
        """
        assert use_annotation in ["label", "skeleton"]
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
                annotations=os.path.join(image_dir, i + f"-{use_annotation}.png"),
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
        ann = skimage.io.imread(annotation_path, 0)
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
        image = cv2.imread(self.image_info[image_id]['path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) 
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        if self.use_rgbd:
            depth = cv2.imread(self.image_info[image_id]['depth'], 0)
            image = np.concatenate((image, np.expand_dims(depth / 10000, 2)), axis=2)
        return image

    def load_pose(self, image_id):
        """
        Load Instance poses for given image_id
        :param image_id: internal ID of the image
        :return:
        poses: A bool array of shape [4, 4, instance count] with
            one pose per instance.
        class_ids: [instance_count] a 1D array of class IDs of the instance masks.
        """
        # If not a YCB image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "YCBV":
            return super(YCBVDataset, self).load_mask(image_id)


        meta_path = self.image_info[image_id]["meta"]
        meta = scio.loadmat(meta_path)
        # pose is saved as an [3, 4, N] matrix; needs to be [4, 4, N]
        poses = meta["poses"]
        # first repeats [0, 0, 0, 1] N times to create an array of
        # shape [1, 4, N] and then concatenates it with the first
        # dimension of the poses matrix to create matrix of shape
        # [4, 4, N] where the last row is always [0, 0, 0, 1]
        fin_pose = np.concatenate((meta["poses"],
                                   np.tile(np.array([[0], [0], [0], [1]]),
                                           (1, 1, meta["poses"].shape[2]))))
        class_ids = np.squeeze(meta["cls_indexes"])
        # IMPORTANT: The indices and poses are not yet in the same order as the masks.
        # This is however important later on, so something needs to be done to achieve
        # this. For now this is handled by load image gt
        return fin_pose, class_ids

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
    IMAGE_RESIZE_MODE = "square"#"none" #
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640
    # FPN_CLASSIF_FC_LAYERS_SIZE = 512
    # TRAIN_BN = None
    # should be half of IMAGE_MAX_DIM I think, because the Anchors are scaled up to twice the scale (?)
    RPN_ANCHOR_SCALES = (20, 40, 80, 160, 320)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 21  # 21 Objects were selected from the original YCB Dataset

    # STEPS_PER_EPOCH = 203
    # TRAIN_ROIS_PER_IMAGE = 70
    USE_DEPTH_AWARE_OPS = False

    LEARNING_RATE = 0.005
    ESTIMATE_6D_POSE = True
    XYZ_MODEL_PATH = "/common/homes/staff/pohl/Code/Python/Mask_RCNN/samples/YCB_Video/XYZ_Models.pkl"

    # RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    # IMAGE_CHANNEL_COUNT = 4
    # BACKBONE = da_resnet_graph
    # COMPUTE_BACKBONE_SHAPE = lambda config, image_shape : np.array(
    #     [[int(math.ceil(image_shape[0] / stride)),
    #       int(math.ceil(image_shape[1] / stride))]
    #      for stride in config.BACKBONE_STRIDES])
    # MAX_GT_INSTANCES = 50 # was 100, but normally there should never be more than 100 GT Instances in on picture

########################################################################################################################
#                                               Image Augmentation                                                     #
########################################################################################################################

import imgaug as ia
from imgaug import augmenters as iaa

# random example images
images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode="constant",
            pad_cval=0
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=0, # if mode is constant, use a cval between 0 and 255
            mode="constant" # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.ContrastNormalization((0.5, 2.0))
                    )
                ]),
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                # iaa.Grayscale(alpha=(0.0, 1.0)), # This thorws and CV2 Error sometimes (always?)
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25,
                                                    mode="constant", cval=0)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05), mode="constant", cval=0)), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        )
    ],
    random_order=True
)

aug = iaa.WithChannels(
    channels=[0, 1, 2],
    children=seq
)


########################################################################################################################
#                                                   Evaluation                                                         #
########################################################################################################################

def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = utils.trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = utils.trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = utils.compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    mask_f1 = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            mask_true_pos = np.sum(np.logical_and(pred_masks[:, :, i], gt_masks[:, :, j]))
            mask_false_neg = np.sum(np.logical_and(np.logical_not(pred_masks[:, :, i]), gt_masks[:, :, j]))
            mask_false_pos = np.sum(np.logical_and(pred_masks[:, :, i], np.logical_not(gt_masks[:, :, j])))
            mask_recall = mask_true_pos / (mask_true_pos + mask_false_neg)
            mask_precision = mask_true_pos / (mask_true_pos +  mask_false_pos)
            mask_f1[j] = mask_precision * mask_recall / (mask_precision + mask_recall)
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps, mask_f1

def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps, mask_f1 = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    # TODO: this is rather an F1 score than an mean average precision?
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps, mask_f1

def evaluate_YCBV(model, dataset, config, eval_type="bbox", limit=0, image_ids=None, plot=False, random=True):
    """Runs YCBV evaluation.
        dataset: A Dataset object with valiadtion data
        eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
        limit: if not 0, it's the number of images to use for evaluation
        """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids
    # classes_dict = load_classes_id_dict()
    if random:
        image_ids = np.random.permutation(image_ids)
    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    t_prediction = {}
    t_start = time.time()

    results = []
    APs, precisions, recalls, overlaps, mask_f1s = {}, {}, {}, {}, {}
    # for i, image_id in enumerate(tqdm(image_ids)):
    for i, image_id in enumerate(image_ids):

        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction[image_id] = (time.time() - t)


        # gt_image, gt_image_meta, gt_class_ids, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id)
        gt_image = dataset.load_image(image_id)
        gt_mask, gt_class_ids = dataset.load_mask(image_id)
        gt_bbox = utils.extract_bboxes(gt_mask)
        try:
            AP, precision, recall, overlap, mask_f1 = \
                compute_ap(gt_bbox, gt_class_ids, gt_mask,
                                 r['rois'], r['class_ids'], r['scores'], r['masks'])
        except IndexError:
            print(f"Skipping Image {image_id} because of IndexError")
            continue
        if plot:
            fig, ax = plt.subplots(1, 2)
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'], ax=ax[0])
            visualize.display_instances(gt_image, gt_bbox, gt_mask, gt_class_ids, dataset.class_names,
                                        ax=ax[1])
            fig.tight_layout(pad=0)
            fig.subplots_adjust(wspace=0)
            plt.show()
            visualize.plot_overlaps(gt_class_ids, r['class_ids'], r['scores'],
                                    overlap, dataset.class_names, threshold=config.DETECTION_MIN_CONFIDENCE)
        APs[image_id] = AP
        precisions[image_id] = precision
        recalls[image_id] = recall
        overlaps[image_id] = overlap
        mask_f1s[image_id] = np.mean(mask_f1[mask_f1 > -1])
    result = pd.DataFrame(data={"mAPs": APs, "mF1s": mask_f1s, "recall": recalls, "precision": precisions,
                                "overlap": overlaps, "times": t_prediction})
    mAP = result["mAPs"].mean()
    mF1 = result["mF1s"].mean()
    print(f"mAP of the class prediction: {mAP}. mF1 of the predicted masks: {mF1}")
    print("Prediction time: {}. Average {}/image".format(
        result["times"].sum(), result["times"].mean()))
    print("Total time: ", time.time() - t_start)
    return result


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
    parser.add_argument('--model', required=False,
                        default=None,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=100,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--depth_aware', required=False, type=str2bool,
                       default=False,
                       metavar="<use-depth-awareness>",
                       help='Train with depth-aware operations')
    parser.add_argument('--debug', required=False,
                        default="false",
                        metavar="<debug>",
                        help='Start a Debug-Session at port 7000')
    parser.add_argument('--continue_training', required=False, type=str2bool,
                        default=False,
                        metavar="<continue_training>",
                        help='If true, continues training; otherwise start at epoch 0!')
    parser.add_argument('--annotation', required=False,
                        default="label",
                        metavar="<which_annotation>",
                        help='Which annotations to use for the image. For now only "label" or "skeleton" are used')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("Depth Awareness: ", args.depth_aware)
    print("Continue Training: ", args.continue_training)
    print("Annotation: ", args.annotation)

    if args.debug.lower() != "false":
        from tensorflow.python import debug as tf_debug
        sess = tf.Session()
        if args.debug == "cli":
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        else:
            sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:7000", send_traceback_and_source_code=False)
        KB.set_session(sess)

    # Configurations
    if args.command == "train":
        config = YCBVConfig()
        if args.depth_aware:
            config.USE_DEPTH_AWARE_OPS = True
            config.MEAN_PIXEL = np.append(config.MEAN_PIXEL, 0.0)
            config.IMAGE_CHANNEL_COUNT = 4
        else:
            assert config.USE_DEPTH_AWARE_OPS == False
    else:
        class InferenceConfig(YCBVConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.9
            USE_DEPTH_AWARE_OPS = args.depth_aware
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
    if args.model is not None:
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
                "mrcnn_bbox", "mrcnn_mask"], continue_training=args.continue_training)
        else:
            model.load_weights(model_path, by_name=True, continue_training=args.continue_training)
    else:
        print("Training from scratch")

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.

        dataset_train = YCBVDataset()
        dataset_train.load_ycbv(args.dataset, "train", use_rgbd=args.depth_aware, use_annotation=args.annotation)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = YCBVDataset()
        val_type = "minival"
        dataset_val.load_ycbv(args.dataset, val_type, use_rgbd=args.depth_aware, use_annotation=args.annotation)
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        # TODO: if image is flipped, the 6d Pose changes --> make amends
        # TODO: check if image augmentation works for rgbd images as well, then create more sophisticated augmentation
        #augmentation = seq
        # augmentation = ia.augmenters.Fliplr(0.5)
        augmentation = None
        # *** This training schedule is an example. Update to your needs ***
        # Training - Stage 1
        num = 0
        print("Training Resnet")
        # layers = "resnet"
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE * 10,
        #             epochs=40,
        #             layers=layers,
        #             augmentation=augmentation)
        # builder = tf.profiler.ProfileOptionBuilder
        # opts = builder(builder.time_and_memory()).order_by('micros').build()
        # opts2 = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        num += 50
        layers = "heads"
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=num,
                    layers=layers,
                    augmentation=augmentation)
        # tl = timeline.Timeline(model.run_metadata.step_stats)
        # ctf = tl.generate_chrome_trace_format()
        #
        # with open('./timeline.json', 'w') as f:
        #     f.write(ctf)
        #
        # ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder
        # opts = ProfileOptionBuilder(ProfileOptionBuilder.time_and_memory()
        #                             ).with_node_names(show_name_regexes=['.*']).build()
        # prof = tf.profiler.profile(KB.get_session().graph, model.run_metadata, cmd="code", options=opts)
        # prof = tf.profiler.Profiler(graph=KB.get_session().graph)
        # prof.add_step(1, model.run_metadata)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        num += 50
        layers = '4+'
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE/5,
                    epochs=num,
                    layers=layers,
                    augmentation=augmentation)
        # prof.add_step(2, model.run_metadata)


        # # Training - Stage 3
        # # Fine tune all layers
        num += 50
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE / 50,
                        epochs=num,
                        layers='all',
                        augmentation=augmentation)
        # prof.add_step(3, model.run_metadata)
        # with open('./profile_rgbd', 'wb') as f:
        #     f.write(prof.serialize_to_string())

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = YCBVDataset()
        val_type = "minival"
        dataset_val.load_ycbv(args.dataset, val_type, use_rgbd=args.depth_aware)
        dataset_val.prepare()
        num = args.limit or len(dataset_val.image_ids)
        print(f"Running YCBV evaluation on {num} images.")
        result = evaluate_YCBV(model, dataset_val, config, "bbox", limit=int(args.limit), plot=False)
        result.to_pickle(model_path[:-3] + "_eval.pkl")
    elif args.command == "blub":
        dataset_train = YCBVDataset()
        dataset_train.load_ycbv(args.dataset, "train", use_rgbd=args.depth_aware)
        dataset_train.prepare()
        fig, ax = plt.subplots(5, 2)
        for j in range(5):
            for i in range(2):
                # ax[0].cla()
                # ax[1].cla()
                # fig, ax = plt.subplots(1, 2)
                # plt.close("all")
                visualize.display_instances(image, bbox, mask, class_ids, dataset_train.class_names, ax=ax[j, i])
                # ax[1].imshow(image[:, :, 3])
        plt.show()

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
