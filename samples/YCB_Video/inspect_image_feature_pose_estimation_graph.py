#!/usr/bin/env python
# coding: utf-8

# ## Mask R-CNN - Inspect Trained Model
#
# Code and visualizations to test, debug, and evaluate the Mask R-CNN model.

# In[1]:


import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import open3d as o3d

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from samples.YCB_Video.Test_Pose_Estimation import calculate_2d_hull_of_pointcloud, load_YCB_meta_infos
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.Chamfer_Distance_Loss import mrcnn_pose_loss_model

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
MODEL_PATH = os.path.join(ROOT_DIR, "weights/mask_rcnn_ycbv_pose_estimation_test.h5")
DEBUG = False
if DEBUG:
    import keras.backend as KB
    from tensorflow.python import debug as tf_debug

    sess = tf.Session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    KB.set_session(sess)

def all_equal(iterator):
  try:
     iterator = iter(iterator)
     first = next(iterator)
     return all(np.array_equal(first, rest) for rest in iterator)
  except StopIteration:
     return True

sys.path.append(os.path.join(ROOT_DIR, "samples/YCB_Video/"))  # To find local version
print(os.path.join(ROOT_DIR, "samples/YCB_Video"))
import samples.YCB_Video.YCB_Video as ycbv

config = ycbv.YCBVConfig()
DATASET_DIR = os.path.join(os.path.expanduser("~"), "Hitachi/YCB_Video_Dataset")  # TODO: enter value here


# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_DEPTH_AWARE_OPS = True
    # DETECTION_MIN_CONFIDENCE = 0.0


config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "training"


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


# Build validation dataset

dataset = ycbv.YCBVDataset()
dataset.load_ycbv(DATASET_DIR, "train", use_rgbd=True)
dataset.prepare()

# Create model in inference mode
model = modellib.MaskRCNN(mode=TEST_MODE, model_dir=MODEL_DIR,
                          config=config)

# Load weights
model.load_weights(MODEL_PATH, by_name=True)

# ## Run Detection

image_id = random.choice(dataset.image_ids)
# image_id = 95506
info = dataset.image_info[image_id]
image_gt = dataset.load_image(image_id)
image, image_meta, gt_class_id, gt_bbox, gt_mask, gt_pose = modellib.load_image_gt(dataset, config, image_id,
                                                                                   use_mini_mask=False)
intrinsic_matrix, classes, depth_factor, rot_trans_mat, vertmap, poses, center = load_YCB_meta_infos(info["id"])

# Run object detection
if TEST_MODE is "inference":
    results = model.detect([image], verbose=1)

    # Display results
    ax = get_ax(1)
    r = results[0]

    visualize.display_instances(image[:, :, :3], r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'], ax=ax,
                                title="Predictions", poses=r["poses"], intrinsic_matrix=intrinsic_matrix)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    activations = model.run_graph([image], [
        ("roi_align_pose_image", model.keras_model.get_layer("roi_align_pose").output[0]),
        ("roi_align_pose_depth", model.keras_model.get_layer("roi_align_pose").output[1]),
        ("rois_trans_deconv", model.keras_model.get_layer("mrcnn_pose_rois_trans_deconv").output),  # for resnet100
        ("rois_trans_conv", model.keras_model.get_layer("mrcnn_pose_rois_trans_conv").output),
        ("detections", model.keras_model.get_layer("mrcnn_detection").output),
        ("pose_conv1", model.keras_model.get_layer("mrcnn_pose_conv1").output),
        ("pose_conv4", model.keras_model.get_layer("mrcnn_pose_conv4").output),
        ("pose_conv5", model.keras_model.get_layer("mrcnn_pose_conv5").output),
        ("trans_conva", model.keras_model.get_layer("mrcnn_pose_trans_conva").output),
        ("trans_convb", model.keras_model.get_layer("mrcnn_pose_trans_convb").output),
        ("trans_squeeze", model.keras_model.get_layer("mrcnn_pose_trans_squeeze").output),
        ("rot_conva", model.keras_model.get_layer("mrcnn_pose_rot_conva").output),
        ("rot_convb", model.keras_model.get_layer("mrcnn_pose_rot_convb").output),
        ("mrcnn_class", model.keras_model.get_layer("mrcnn_class").output)
    ])
    det_class_ids = activations['detections'][0, :, 4].astype(np.int32)
    class_logits = activations['detections'][0, :, 5].astype(np.int32)

if TEST_MODE is "training":
    activations = model.run_trainings_graph(dataset, image_id, [
        ("roi_align_pose_image", model.keras_model.get_layer("roi_align_pose").output[0]),
        ("roi_align_pose_depth", model.keras_model.get_layer("roi_align_pose").output[1]),
        ("rois_trans_deconv", model.keras_model.get_layer("mrcnn_pose_rois_trans_deconv").output),  # for resnet100
        ("rois_trans_conv", model.keras_model.get_layer("mrcnn_pose_rois_trans_conv").output),
        ("detections", model.keras_model.get_layer("proposal_targets").output[0]),
        ("target_class_ids", model.keras_model.get_layer("proposal_targets").output[1]),
        ("target_poses", model.keras_model.get_layer("proposal_targets").output[4]),
        ("pose_conv1", model.keras_model.get_layer("mrcnn_pose_conv1").output),
        ("pose_conv4", model.keras_model.get_layer("mrcnn_pose_conv4").output),
        ("pose_conv5", model.keras_model.get_layer("mrcnn_pose_conv5").output),
        ("trans_conva", model.keras_model.get_layer("mrcnn_pose_trans_conva").output),
        ("trans_convb", model.keras_model.get_layer("mrcnn_pose_trans_convb").output),
        ("trans_squeeze", model.keras_model.get_layer("mrcnn_pose_trans_squeeze").output),
        ("rot_conva", model.keras_model.get_layer("mrcnn_pose_rot_conva").output),
        ("rot_convb", model.keras_model.get_layer("mrcnn_pose_rot_convb").output),
        ("mrcnn_class", model.keras_model.get_layer("mrcnn_class").output),
        ("mrcnn_bbox", model.keras_model.get_layer("mrcnn_bbox").output),
        ("mrcnn_class_logits", model.keras_model.get_layer("mrcnn_class_logits").output),
        # ("mrcnn_pose_loss", model.keras_model.get_layer("mrcnn_pose_loss").output),
        ########### from function - mrcnn_pose_loss_graph_keras ###########
        # ("mrcnn_pose_loss", model.keras_model.get_layer("mrcnn_pose_loss/loss").output),
        ("pose_target_class_ids", model.keras_model.get_layer("mrcnn_pose_loss/target_class_ids").output),
        ("pose_target_poses", model.keras_model.get_layer("mrcnn_pose_loss/target_poses").output),
        ("pose_target_trans", model.keras_model.get_layer("mrcnn_pose_loss/target_trans").output),
        ("pose_target_rot", model.keras_model.get_layer("mrcnn_pose_loss/target_rot").output),
        ("pose_pred_trans", model.keras_model.get_layer("mrcnn_pose_loss/pred_trans_transposed").output),
        ("pose_pred_rot", model.keras_model.get_layer("mrcnn_pose_loss/pred_rot_transposed").output),
        ("pose_positive_ix", model.keras_model.get_layer("mrcnn_pose_loss/positive_ix").output),
        ("pose_positive_class_ids", model.keras_model.get_layer("mrcnn_pose_loss/positive_class_ids").output),
        ("pose_indices", model.keras_model.get_layer("mrcnn_pose_loss/indices").output),
        ("pose_y_true_t", model.keras_model.get_layer("mrcnn_pose_loss/y_true_t").output),
        ("pose_y_true_r", model.keras_model.get_layer("mrcnn_pose_loss/y_true_r").output),
        ("pose_y_pred_t", model.keras_model.get_layer("mrcnn_pose_loss/y_pred_t").output),
        ("pose_y_pred_r", model.keras_model.get_layer("mrcnn_pose_loss/y_pred_r").output),
        ("pose_pred_rot_svd_matmul", model.keras_model.get_layer("mrcnn_pose_loss/pred_rot_svd_matmul").output),
        ("pose_pos_xyz_models", model.keras_model.get_layer("mrcnn_pose_loss/pos_xyz_models").output),
        ########### from function - chamfer_distance_loss_keras ###########
        ("transposed_pred_models", model.keras_model.get_layer("transposed_pred_models").output),
        ("total_number_of_points", model.keras_model.get_layer("total_number_of_points").output),
        ("added_pred_models", model.keras_model.get_layer("added_pred_models").output),
        ("transposed_target_models", model.keras_model.get_layer("transposed_target_models").output),
        ("added_target_models", model.keras_model.get_layer("added_target_models").output),
        ("NNDistance1", model.keras_model.get_layer("NNDistance").output[0]),
        ("NNDistance2", model.keras_model.get_layer("NNDistance").output[2]),
        ("reduced_sum1", model.keras_model.get_layer("reduced_sum1").output),
        ("reduced_sum2", model.keras_model.get_layer("reduced_sum2").output),
        ("added_reduced_sum", model.keras_model.get_layer("added_reduced_sum").output),
        ("chamfer_loss", model.keras_model.get_layer("mrcnn_chamfer_loss").output)
    ])
    det_class_ids = activations['target_class_ids'][0].astype(np.int32)

det_count = np.where(det_class_ids == 0)[0][0]
det_class_ids = det_class_ids[:det_count]
detections = activations['detections'][0, :det_count]
# class_logits = class_logits[:det_count]
roi_class_ids = np.argmax(activations["mrcnn_class"][0], axis=1)
roi_scores = activations["mrcnn_class"][0, np.arange(roi_class_ids.shape[0]), roi_class_ids]
roi_class_names = np.array(dataset.class_names)[roi_class_ids]
roi_positive_ixs = np.where(roi_class_ids > 0)[0]

# TODO: I'm not sure if roi_scores[:det_count] is the same as detections[:, 5]
captions = ["{} {:.3f}".format(dataset.class_names[int(c)], s) if c > 0 else ""
            for c, s in zip(det_class_ids, roi_scores[:det_count])]
visualize.draw_boxes(
    image[:, :, :3],
    refined_boxes=utils.denorm_boxes(detections[:, :4], image.shape[:2]),
    visibilities=[2] * len(detections), title="Detections", captions=captions,
    ax=get_ax())

roi = 0
fig, axes = plt.subplots(5, 5)
for i, ax in enumerate(fig.axes):
    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)
    if not i:
        ax.imshow(activations["roi_align_pose_depth"][0, roi, :, :, 0])
    else:
        ax.imshow(activations["roi_align_pose_image"][0, roi, :, :, i - 1])

intrinsic_matrix, classes, depth_factor, rot_trans_mat, vertmap, poses, center = load_YCB_meta_infos(info["id"])
# fig, axes = plt.subplots(1, 2)
# visualize.draw_boxes(image[:, :, :3], utils.denorm_boxes(detections[:, :4], image.shape[:2]),  ax=axes[0])

# translations computed from the rois [roi, 3]
roi_translations = np.squeeze(activations["rois_trans_conv"][0, :det_count], axis=2)
trans_list = [roi_translations.transpose(0, 2, 1)[i, det_class_ids[i], :].reshape(1, -1) for i in
              range(len(det_class_ids))]
roi_translations = np.concatenate(trans_list, axis=0)

# translations computed from convolutions [roi, 3]
conv_translations = np.squeeze(activations["trans_convb"][0, :det_count], axis=2)
trans_list = [conv_translations.transpose(0, 2, 1)[i, det_class_ids[i], :].reshape(1, -1) for i in
              range(len(det_class_ids))]
conv_translations = np.concatenate(trans_list, axis=0)

fin_translations = activations["trans_squeeze"][0, :det_count]
trans_list = [fin_translations.transpose(0, 2, 1)[i, det_class_ids[i], :].reshape(1, -1) for i in
              range(len(det_class_ids))]
fin_translations = np.concatenate(trans_list, axis=0)

translations = roi_translations + conv_translations
assert np.array_equal(fin_translations, translations)

rotations = activations["rot_convb"][0, :det_count]
rot_list = [np.expand_dims(rotations.transpose(0, 3, 1, 2)[i, det_class_ids[i], :],
                           axis=0) for i in range(len(det_class_ids))]
rotations = np.concatenate(rot_list, axis=0)

with open(config.XYZ_MODEL_PATH, "rb") as f:
    df = np.array(pkl.load(f), dtype=np.float32)
xyz_models = np.transpose(df, (0, 2, 1))
num_xyz_points = xyz_models.shape[2]
# pose_loss_model = mrcnn_pose_loss_model(config.NUM_CLASSES, config.TRAIN_ROIS_PER_IMAGE, xyz_models.shape[2])
# target_trans, target_rot, pred_trans, pred_rot, positive_ix, positive_class_ids, indices, y_true_t, y_true_r, y_pred_t, y_pred_r, pos_xyz_models, loss = pose_loss_model.predict_on_batch(
#     [activations["target_poses"], activations["target_class_ids"], activations["trans_squeeze"],
#      activations["rot_convb"], xyz_models])
##### test for correct shapes ####
batch_times_num_rois = activations["pose_target_class_ids"].shape
num_positive_ix = activations["pose_positive_ix"].shape
assert activations["pose_target_trans"].shape == batch_times_num_rois + (1, 3,)
assert activations["pose_target_rot"].shape == batch_times_num_rois + (3, 3,)
assert activations["pose_pred_trans"].shape == batch_times_num_rois + (config.NUM_CLASSES, 1, 3,)
assert activations["pose_pred_rot"].shape == batch_times_num_rois + (config.NUM_CLASSES, 3, 3,)
assert activations["pose_positive_class_ids"].shape == num_positive_ix
assert activations["pose_indices"].shape == num_positive_ix + (2, )
assert activations["pose_y_true_t"].shape == num_positive_ix + (1, 3,)
assert activations["pose_y_true_r"].shape == num_positive_ix + (3, 3,)
assert activations["pose_y_pred_t"].shape == num_positive_ix + (1, 3,)
assert activations["pose_y_pred_r"].shape == num_positive_ix + (3, 3,)
assert activations["pose_pos_xyz_models"].shape == num_positive_ix + (3, num_xyz_points)
assert activations["transposed_pred_models"].shape == num_positive_ix + (num_xyz_points, 3)
assert activations["added_pred_models"].shape == num_positive_ix + (num_xyz_points, 3)
assert activations["transposed_target_models"].shape == num_positive_ix + (num_xyz_points, 3)
assert activations["added_target_models"].shape == num_positive_ix + (num_xyz_points, 3)
assert activations["NNDistance1"].shape == num_positive_ix + (num_xyz_points, )
assert activations["NNDistance2"].shape == num_positive_ix + (num_xyz_points, )
assert activations["reduced_sum1"].shape == tuple()
assert activations["reduced_sum2"].shape == tuple()
#### equivalency tests

assert np.array_equal(activations["pose_target_class_ids"][:det_count], det_class_ids)
assert np.array_equal(np.where(activations["target_class_ids"][0] > 0)[0], activations["pose_positive_ix"])
# y_true_r is equal to the rotation of the object in the meta-file plus the changes made during rescaling
# check that the "true" poses are the correct ones for the objects inside the rois
index = np.searchsorted(classes.reshape(-1), det_class_ids, sorter=np.argsort(classes.reshape(-1)))
det_class_indeces = np.take(np.argsort(classes.reshape(-1)), index, mode="clip")
# TODO: this only works for this specific dataset were the pad
offset = np.array([0, 80, 0])
camera_offset = np.expand_dims(np.matmul(np.linalg.inv(intrinsic_matrix), offset), axis=-1)
# for i in range(det_count):
#     corresponding_pose = poses[:, :, det_class_indeces[i]].astype("float32")
#     corresponding_pose[:3, 3] = corresponding_pose[:3, 3] + camera_offset[:, 0]
#     assert classes[det_class_indeces[i]][0] == activations["pose_target_class_ids"][i]
#     np.testing.assert_allclose(activations["pose_y_true_r"][i], corresponding_pose[:3, :3], rtol=1e-5)
#     np.testing.assert_allclose(activations["pose_y_true_t"][i],
#                                np.expand_dims(corresponding_pose[:3, 3], axis=0), rtol=1e-5)
#     concat_pose = np.concatenate([activations["pose_y_true_r"][i],
#                                   np.transpose(activations["pose_y_true_t"], [0, 2, 1])[i]], axis=1)
#     np.testing.assert_allclose(concat_pose, corresponding_pose, rtol=1e-5)
# check that the selected point clouds are the correct ones for the object inside the roi
# check that rois with the same gt_id have selected the same point_cloud
for i in np.unique(det_class_ids):
    ids = np.where(det_class_ids == i)[0]
    assert all_equal(activations["pose_pos_xyz_models"][ids])
#TODO: check that the poses and translations and their groundtruths are on the same scale (rel. image, vs world?)

# transl is [n, 1, 3] and needs to be [N, 3, 1] to concatenate to [N, 3, 4] poses
concat_poses = np.concatenate([activations["pose_y_true_r"], np.transpose(activations["pose_y_true_t"], [0, 2, 1])], axis=2)
visualize.visualize_poses(image, concat_poses, activations["pose_positive_class_ids"], intrinsic_matrix)
visualize.visualize_poses(image_gt, poses.transpose(2, 0, 1), classes, intrinsic_matrix)
models = np.transpose(activations["pose_pos_xyz_models"], [0, 2, 1])
homogeneous_models = np.concatenate([models, np.tile([1], (models.shape[0], models.shape[1], 1))], axis=2)
trans_hom_models = np.matmul(concat_poses, np.transpose(homogeneous_models, [0, 2, 1])).transpose([0, 2, 1])
np.testing.assert_allclose(trans_hom_models, activations["added_target_models"], rtol=1e-3)
identity_poses = np.zeros_like(concat_poses)
identity_poses[:, 0, 0] = 1
identity_poses[:, 1, 1] = 1
identity_poses[:, 2, 2] = 1

visualize.visualize_pointcloud_hulls(image, concat_poses, models,
                                     activations["pose_positive_class_ids"], intrinsic_matrix)
# visualize the target models, that are calculated using the gt_poses in the chamfer loss function
visualize.visualize_pointcloud_hulls(image, identity_poses, activations["added_target_models"],
                                     activations["pose_positive_class_ids"], intrinsic_matrix)


# check that the svd was done the same way as in numpy
u, s, vh = np.linalg.svd(activations["pose_y_pred_r"])
r = np.matmul(u, vh)
# TODO: this should be the same somehow, but tf return the adjoint of v and np does not
# still they should be the same if tf.linalg.matmul(u,v, adjoint_b=True) is used, but its not
# np.testing.assert_allclose(r, activations["pose_pred_rot_svd_matmul"], rtol=1e-3)
concat_poses = np.concatenate([activations["pose_pred_rot_svd_matmul"],
                               np.transpose(activations["pose_y_pred_t"], [0, 2, 1])], axis=2)
visualize.visualize_poses(image, concat_poses, activations["pose_positive_class_ids"], intrinsic_matrix)
target_pc = o3d.PointCloud()
target_pc.points = o3d.Vector3dVector(activations["added_target_models"].reshape(-1, 3))
pred_pc = o3d.PointCloud()
pred_pc.points = o3d.Vector3dVector(activations["added_pred_models"].reshape(-1, 3))
o3d.draw_geometries([pred_pc, target_pc])