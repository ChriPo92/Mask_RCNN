import os
import os.path as osp
import sys
import random
import numpy as np
import scipy.io as io
import subprocess as sp
import open3d as o3d
from samples.YCB_Video.YCB_Video import YCBVConfig, YCBVDataset
from samples.YCB_Video.open3d_test import preprocess_point_cloud,\
    execute_global_registration, refine_registration
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import ConvexHull
from shutil import copyfile

print(os.environ["LD_LIBRARY_PATH"])
# Root directory of the project
ROOT_DIR = os.path.abspath("../")
print(ROOT_DIR)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from samples.LINEMOD.LINEMOD import linemod_point_cloud
import mrcnn.model as modellib
from mrcnn.data_generation import load_image_gt
from mrcnn import visualize

import obj_pose_eval.renderer as renderer
import obj_pose_eval.inout as inout
import obj_pose_eval.pose_error as pose_error
import obj_pose_eval.transform as transform

# DATA_ROOT_PATH = os.path.expanduser("~")
# DATA_ROOT_PATH = /media/pohl
DATA_ROOT_PATH = os.path.expanduser("~/Data/YCB_Video_Dataset/")
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

# from keras.backend.tensorflow_backend import set_session
# import tensorflow as tf
# from tensorflow.python import debug as tf_debug
# sess = tf.Session()
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, "i61pc031:7000")
# set_session(sess)
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# set_session(tf.Session(config=config))


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "weights/mask_rcnn_ycbv_rgbd_custom_da_resnet_graph.h5")
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/ycbv20181212T1038/mask_rcnn_ycbv_0180.h5")
# COCO_MODEL_PATH = "/common/homes/staff/pohl/Code/Python/Mask_RCNN/logs/ycbv20181212T1038/mask_rcnn_ycbv_0180.h5"
COCO_MODEL_PATH = os.path.join(os.path.expanduser("~"), "Code/Python/Mask_RCNN/weights/mask_rcnn_ycbv_pose_estimation_test_1.h5")
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "weights/mask_rcnn_ycbv_rgbd.h5")
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# class InferenceConfig(coco.CocoConfig):
#     # Set batch size to 1 since we'll be running inference on
#     # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
#     DETECTION_MIN_CONFIDENCE = 0.4
#     USE_DEPTH_AWARE_OPS = False
#     IMAGE_MIN_DIM = 480
#     IMAGE_MAX_DIM = 640
#     NUM_CLASSES = 1 + 21

class InferenceConfig(YCBVConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_DEPTH_AWARE_OPS = True
    POSE_ESTIMATION_METHOD = "pointnet"
    POINTNET_VECTOR_SIZE = 256
    TRAIN_ROIS_PER_IMAGE = 100
    DETECTION_MAX_INSTANCES = 100
    # IMAGE_RESIZE_MODE = "square"


config = InferenceConfig()
# config.USE_DEPTH_AWARE_OPS = True
# config.MEAN_PIXEL = np.append(config.MEAN_PIXEL, 0.0)
# config.IMAGE_CHANNEL_COUNT = 4
# config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# model.compile(0.1, 0.9)
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def calculate_2d_hull_of_pointcloud(pc, rot, trans, camera_calibration_matrix):
    T_o_c = np.zeros((4, 4))
    T_o_c[:3, :3] = np.array(rot).reshape(3, 3)
    T_o_c[:3, 3] = np.array(trans).reshape(3, )
    T_o_c[3, 3] = 1
    hom_pc = cv2.convertPointsToHomogeneous(pc).reshape(-1, 4).transpose()
    pc_cam_hom = np.matmul(T_o_c, hom_pc).transpose().reshape(-1, 1, 4)
    pc_cam = cv2.convertPointsFromHomogeneous(pc_cam_hom).reshape(-1, 3)
    pc_2d, _ = cv2.projectPoints(pc_cam, np.array([0, 0, 0], np.float32), np.array([0, 0, 0], np.float32),
                                 camera_calibration_matrix, 0)
    pc_2d = pc_2d.reshape(-1, 2)
    hull = ConvexHull(pc_2d)
    return pc_2d, hull


def load_YCB_meta_infos(id):
    path = DATA_ROOT_PATH + "data/%s-meta.mat" % id
    meta = io.loadmat(path)
    int_matrix = meta["intrinsic_matrix"]
    classes = meta["cls_indexes"]
    depth_factor = meta["factor_depth"]
    rot_trans_mat = meta["rotation_translation_matrix"]
    vertmap = meta["vertmap"]
    poses = meta["poses"]
    center = meta["center"]
    return int_matrix, classes, depth_factor, rot_trans_mat, vertmap, poses, center


def get_orientation_line_points(pose, K, scale=0.05):
    """
    calculates the start and end points of each principal vector along the unit axis in the object frame in image coordinates
    :return:
    """
    # coordinates in homogeneous coordinates in the object frame
    origin = np.array([0, 0, 0, 1])
    x_axis = np.array([scale, 0, 0, 1])
    y_axis = np.array([0, scale, 0, 1])
    z_axis = np.array([0, 0, scale, 1])
    T_o_c = pose
    origin_cam = np.matmul(T_o_c, origin)
    x_axis_cam = np.matmul(T_o_c, x_axis)
    y_axis_cam = np.matmul(T_o_c, y_axis)
    z_axis_cam = np.matmul(T_o_c, z_axis)
    # cam_rot = RT[:, :3].copy()
    # rvec, _ = cv2.Rodrigues(cam_rot)
    # tvec = RT[:, 3:].copy()
    rvec = np.array([0, 0, 0], np.float32)
    tvec = rvec.copy()  # 6D Poses are given in Camera Coordinates and not in World Coordinates
    dist_coeffs = 0  # np.array([0.04112172, -0.4798174, 0, 0, 1.890084])
    coords = np.concatenate((origin_cam[:3], x_axis_cam[:3], y_axis_cam[:3], z_axis_cam[:3])).reshape(-1, 3)
    image_points, _ = cv2.projectPoints(coords, rvec, tvec, K, dist_coeffs)
    return image_points.reshape(-1, 2)


def load_bbox(id):
    path = DATA_ROOT_PATH + "data/%s-box.txt" % id
    d = {}
    with open(path, "r") as f:
        for row in f:
            splt = row.split(" ")
            # [[y_0, x_0], [y_1, x_1]] same as the rois in Mask RCNN results
            d[splt[0]] = np.array([float(splt[2]), float(splt[1]), float(splt[4]), float(splt[3])])
    return d


def load_classes_id_dict():
    path = DATA_ROOT_PATH + "image_sets/classes.txt"
    d = {}
    with open(path, "r") as f:
        for i, val in enumerate(f):
            d[i +1] = val[:-1]
    return d


dataset = YCBVDataset()
dataset.load_ycbv(DATA_ROOT_PATH, "trainval", use_rgbd=config.USE_DEPTH_AWARE_OPS)
dataset.prepare()

dpt_file = DATA_ROOT_PATH + "data/0054/000016-depth.png"
img_file = DATA_ROOT_PATH + "data/0054/000016-color.png"
# img_file = "/media/pohl/Hitachi/YCB_Video_Dataset/data/0081/000982-color.png"
# dpt_file = "/media/pohl/Hitachi/YCB_Video_Dataset/data/0054/000001-depth.png"
# dpt_file = "/common/homes/staff/pohl/Code/Python/Mask_RCNN/images/RobDekon/snapshot_11-22-2018_11-35-34.152_depth.png"
# img_file = "/common/homes/staff/pohl/Code/Python/Mask_RCNN/images/RobDekon/snapshot_11-22-2018_11-35-34.152_rgb.png"
import skimage.io as skio

image_id = random.choice(dataset.image_ids)
# image_id = 8736
image_molded, image_meta, gt_class_id, gt_bbox, gt_mask, gt_pose, intrinsic_matrix_gt =\
    load_image_gt(dataset, config, image_id, use_mini_mask=False)
image = dataset.load_image(image_id)
depth = image[:, :, 3]
info = dataset.image_info[image_id]

intrinsic_matrix, classes, depth_factor, rot_trans_mat, vertmap, poses, center = load_YCB_meta_infos(info["id"])
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                       dataset.image_reference(image_id)))
# Run object detection
results = model.detect([image], verbose=1, intrinsic_matrix=[intrinsic_matrix])

# X = []
# for i in range(len(classes)):
#     points = get_orientation_line_points(poses[:, :, i], intrinsic_matrix)
#     X.append(points)
# mug_pose = poses[:, :, 1]
# pc_2d, hull = calculate_2d_hull_of_pointcloud(pc, mug_pose[:, :3], mug_pose[:, 3], intrinsic_matrix)

# fig, ax = plt.subplots(1, 2)
# fig.set_size_inches(15, 30)
# ax[0].imshow(image)
# # ax[0].plot(center[:, 0], center[:, 1], "r+")
# # for arr in X:
# #     ax[0].plot([arr[0, 0], arr[1, 0]], [arr[0, 1], arr[1, 1]], "r-")
# #     ax[0].plot([arr[0, 0], arr[2, 0]], [arr[0, 1], arr[2, 1]], "g-")
# #     ax[0].plot([arr[0, 0], arr[3, 0]], [arr[0, 1], arr[3, 1]], "b-")
# # from matplotlib.patches import Rectangle
# # for key, val in bboxs.items():
# #     rect = Rectangle(val[0], val[1][0]-val[0][0], val[1][1]-val[0][1], edgecolor="r", facecolor="none")
# #     ax[0].add_patch(rect)
# # # ax[0].plot(pc_2d[:, 0], pc_2d[:, 1])
# # for simplex in hull.simplices:
# #     ax[0].plot(pc_2d[simplex, 0], pc_2d[simplex, 1], 'k-')
# ax[1].imshow(depth)
# ax[1].plot(center[:, 0], center[:, 1], "r+")

classes_dict = load_classes_id_dict()


# image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(dataset, config, dataset.image_from_source_map["YCBV.0054/000016"])
# Run detection
# results = model.detect([np.concatenate((image, np.expand_dims(depth, 2)), axis=2)], verbose=1)
# results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            classes_dict, r['scores'], poses=r["poses"],
                            intrinsic_matrix=intrinsic_matrix)
# visualize.display_differences(image, bbox, class_ids, mask, r['rois'], r['class_ids'], r['scores'], r['masks'], classes_dict)


#
#
# row, col = np.where(r["masks"][:, :, 1])
# pos = np.concatenate((col.reshape(-1, 1), row.reshape(-1, 1)), axis=1)
# hom_pos = cv2.convertPointsToHomogeneous(pos).reshape(-1, 3).transpose()
#
# print(hom_pos)
# cam_pos = np.matmul(np.linalg.inv(intrinsic_matrix), hom_pos).transpose()
# xyz = np.multiply(cam_pos, depth[r["masks"][:, :, 1]].reshape(-1, 1) / 10000)
# xyz = xyz[np.abs(xyz[:, 0]) > 0.000001]
# xyz = xyz[np.abs(xyz[:, 2] - np.median(xyz[:, 2])) < np.std(xyz[:, 2])]
# mean = np.mean(xyz, axis=0)
# fig = plt.figure()
# fig.set_size_inches(10, 20)
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])
# ax.view_init(90, 90)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# with open("test.xyz", "w") as f:
#     for row in xyz:
#         # f.write("%.5f %.5f %.5f\n"%(row[0] - mean[0], row[1] - mean[1], row[2] - mean[2]))
#         f.write("%.5f %.5f %.5f\n"%(row[0], row[1], row[2]))

def get_icp_RT(results, bboxes, intrinsic_matrix):
    rois = results["rois"]
    item_correspondences = {}
    poses = {}
    # find the corresponding mask detected by Mask RCNN for the object
    for key, value in bboxes.items():
        item_correspondences[key] = np.argmin(np.sum(np.abs(rois - value), axis=1))
    for key, value in item_correspondences.items():
        # extract y and x values of the mask
        row, col = np.where(results["masks"][:, :, value])
        # create array [x, y]
        pos = np.concatenate((col.reshape(-1, 1), row.reshape(-1, 1)), axis=1)
        # transform to homogenous coordinates
        hom_pos = cv2.convertPointsToHomogeneous(pos).reshape(-1, 3).transpose()
        # go from image coordinates to relative (to z) camera coordinates
        cam_pos = np.matmul(np.linalg.inv(intrinsic_matrix), hom_pos).transpose()
        # get absolute camera coordinates
        xyz = np.multiply(cam_pos, depth[results["masks"][:, :, value]].reshape(-1, 1) / 10000)
        # approximately calculate the center of the of masked points by calculating the mean
        # mean = np.mean(xyz, axis=0)
        # discard values with depth = 0
        xyz = xyz[np.abs(xyz[:, 0]) > 0.000001]
        # filter out some outliers in z
        xyz = xyz[np.abs(xyz[:, 2] - np.median(xyz[:, 2])) < np.std(xyz[:, 2])]
        if osp.exists("mask.xyz"):
            os.remove("mask.xyz")
        with open("mask.xyz", "w") as f:
            for row in xyz:
                # f.write("%.5f %.5f %.5f\n"%(row[0] - mean[0], row[1] - mean[1], row[2] - mean[2]))
                f.write("%.5f %.5f %.5f\n" % (row[0], row[1], row[2]))
        if osp.exists("mask.pcd"):
            os.remove("mask.pcd")
        if osp.exists("model.pcd"):
            os.remove("model.pcd")
        # transform xyz pointcloud to pcl format
        sp.run(["pcl_xyz2pcd", "mask.xyz", "mask.pcd"], stdout=sp.DEVNULL, check=True)
        model_path = osp.join(DATA_ROOT_PATH + "models", key)
        if not osp.exists(osp.join(model_path, "model_downsampled.pcd")):
            sp.run(["pcl_obj2pcd", osp.join(model_path, "textured.obj"), osp.join(model_path, "model.pcd")],
                   stdout=sp.DEVNULL, check=True)
            sp.run(["pcl_uniform_sampling", osp.join(model_path, "model.pcd"),
                    osp.join(model_path, "model_downsampled.pcd"), "-radius", "0.002"], stdout=sp.DEVNULL, check=True)
        # sp.run(["pcl_xyz2pcd", osp.join(model_path, "points.xyz"), "model.pcd"], stdout=sp.DEVNULL, check=True)
        copyfile(osp.join(model_path, "model_downsampled.pcd"), "model.pcd")
        completed_process = sp.run(["pcl_icp", "-d", "1.0", "model.pcd", "mask.pcd"], stdout=sp.PIPE, check=True)
        str_arr = str(completed_process.stdout).split("\\n")[6:10]
        # sp.run(["pcl_viewer", "model.pcd", "mask.pcd"])
        pose = np.array([np.array(row.split(), np.float32) for row in str_arr])
        # we have to add the previously subtracted mean
        # pose[:3, 3] += mean
        # it turns out that icp becomes more precise if it is done with 2 different scales for MaxCorrespondenceDistance
        completed_process = sp.run(["pcl_icp", "-d", "0.01", "-r", "0.01", "-i", "100", "model.pcd", "mask.pcd"],
                                   stdout=sp.PIPE, check=True)
        str_arr = str(completed_process.stdout).split("\\n")[6:10]
        # sp.run(["pcl_viewer", "model.pcd", "mask.pcd"])
        pose2 = np.array([np.array(row.split(), np.float32) for row in str_arr])
        # pose[:3, :3] = np.matmul(pose2[:3, :3], pose[:3, :3])
        # pose[:3, 3] += pose2[:3, 3]
        poses[key] = np.matmul(pose2, pose)
    return poses, item_correspondences

def get_ransac_RT(results, bboxes, intrinsic_matrix, depth, image, voxel_size=0.005):
    rois = results["rois"]
    item_correspondences = {}
    poses = {}
    # find the corresponding mask detected by Mask RCNN for the object
    for key, value in bboxes.items():
        item_correspondences[key] = np.argmin(np.sum(np.abs(rois - value), axis=1))
    for key, value in item_correspondences.items():
        # extract y and x values of the mask
        row, col = np.where(results["masks"][:, :, value])
        # create array [x, y]
        pos = np.concatenate((col.reshape(-1, 1), row.reshape(-1, 1)), axis=1)
        # transform to homogenous coordinates
        hom_pos = cv2.convertPointsToHomogeneous(pos).reshape(-1, 3).transpose()
        # go from image coordinates to relative (to z) camera coordinates
        cam_pos = np.matmul(np.linalg.inv(intrinsic_matrix), hom_pos).transpose()
        # get absolute camera coordinates
        xyz = np.multiply(cam_pos, depth[results["masks"][:, :, value]].reshape(-1, 1) / 10000)
        rgb = image[results["masks"][:, :, value]].reshape(-1, 3)
        # approximately calculate the center of the of masked points by calculating the mean
        # mean = np.mean(xyz, axis=0)
        # discard values with depth = 0
        m1 = np.abs(xyz[:, 0]) > 0.000001
        xyz = xyz[m1]
        rgb = rgb[m1]
        # filter out some outliers in z
        # TODO: use outlier filtering from o3d
        m2 = np.abs(xyz[:, 2] - np.median(xyz[:, 2])) < np.std(xyz[:, 2])
        xyz = xyz[m2]
        rgb = rgb[m2]
        # if osp.exists("mask.xyz"):
        #     os.remove("mask.xyz")
        # with open("mask.xyz", "w") as f:
        #     for row in xyz:
        #         # f.write("%.5f %.5f %.5f\n"%(row[0] - mean[0], row[1] - mean[1], row[2] - mean[2]))
        #         f.write("%.5f %.5f %.5f\n" % (row[0], row[1], row[2]))
        # if osp.exists("mask.pcd"):
        #     os.remove("mask.pcd")
        # if osp.exists("model.pcd"):
        #     os.remove("model.pcd")
        # transform xyz pointcloud to pcl format
        # sp.run(["pcl_xyz2pcd", "mask.xyz", "mask.pcd"], stdout=sp.DEVNULL, check=True)

        model_path = osp.join(DATA_ROOT_PATH + "models", key)
        if not osp.exists(osp.join(model_path, "model.pcd")):
            sp.run(["pcl_obj2pcd", osp.join(model_path, "textured.obj"), osp.join(model_path, "model.pcd")],
                   stdout=sp.DEVNULL, check=True)
        ycbv_pcd = o3d.read_point_cloud(osp.join(model_path, "model.pcd"))
        mrcnn_pcd = o3d.PointCloud()
#        o3d.estimate_normals(mrcnn_pcd)
 #       o3d.estimate_normals(ycbv_pcd)
        mrcnn_pcd.points = o3d.Vector3dVector(xyz)
        # mrcnn_pcd.colors = o3d.Vector3dVector(rgb / 255)
        ycbv_pcd_down, ycbv_fpfh = preprocess_point_cloud(ycbv_pcd, voxel_size)
        mrcnn_pcd_down, mrcnn_fpfh = preprocess_point_cloud(mrcnn_pcd, voxel_size)
        result_ransac = execute_global_registration(ycbv_pcd_down, mrcnn_pcd_down,
                                                    ycbv_fpfh, mrcnn_fpfh, voxel_size)
        # draw_registration_result(ycbv_pcd_down, mrcnn_pcd_down,
        #                          result_ransac.transformation)
        ycbv_pcd, _ = preprocess_point_cloud(ycbv_pcd, voxel_size / 3)
        mrcnn_pcd, _ = preprocess_point_cloud(mrcnn_pcd, voxel_size / 3)
        result_icp = refine_registration( ycbv_pcd, mrcnn_pcd, voxel_size, result_ransac)

            # sp.run(["pcl_uniform_sampling", osp.join(model_path, "model.pcd"),
            #         osp.join(model_path, "model_downsampled.pcd"), "-radius", "0.002"], stdout=sp.DEVNULL, check=True)
        # sp.run(["pcl_xyz2pcd", osp.join(model_path, "points.xyz"), "model.pcd"], stdout=sp.DEVNULL, check=True)
        # copyfile(osp.join(model_path, "model_downsampled.pcd"), "model.pcd")
        # completed_process = sp.run(["pcl_icp", "-d", "1.0", "model.pcd", "mask.pcd"], stdout=sp.PIPE, check=True)
        # str_arr = str(completed_process.stdout).split("\\n")[6:10]
        # sp.run(["pcl_viewer", "model.pcd", "mask.pcd"])
        # pose = np.array([np.array(row.split(), np.float32) for row in str_arr])
        # we have to add the previously subtracted mean
        # pose[:3, 3] += mean
        # it turns out that icp becomes more precise if it is done with 2 different scales for MaxCorrespondenceDistance
        # completed_process = sp.run(["pcl_icp", "-d", "0.01", "-r", "0.01", "-i", "100", "model.pcd", "mask.pcd"],
        #                            stdout=sp.PIPE, check=True)
        # str_arr = str(completed_process.stdout).split("\\n")[6:10]
        # # sp.run(["pcl_viewer", "model.pcd", "mask.pcd"])
        # pose2 = np.array([np.array(row.split(), np.float32) for row in str_arr])
        # pose[:3, :3] = np.matmul(pose2[:3, :3], pose[:3, :3])
        # pose[:3, 3] += pose2[:3, 3]
        poses[key] = result_icp.transformation
    return poses, item_correspondences


def visualize_icp_vs_ground_truth(image, depth, poses, classes, gt_poses, gt_classes, classes_dict, intrinsic_matrix,
                                  masks=None, item_corr=None, mode="vector", show_mask=False):
    assert mode in ["vector", "hull", "both"]
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(15, 30)
    ax[0].imshow(image)
    ax[1].imshow(depth)
    f, a = plt.subplots(len(gt_classes), 2)
    errors = []
    rgb, dep = [], []
    for i, id in enumerate(gt_classes):
        name = classes_dict[id[0]]  # id = 0 is probably background ??
        gt_pose = gt_poses[:, :, i]
        gt_pose_dict = {"R": gt_pose[:3, :3], "t": gt_pose[:3, 3:] * 1000}
        try:
            j = np.where(classes == id[0])[0][0]
        except IndexError:
            continue
        pose = poses[j]
        assert pose.shape == (4, 4)
        pose_dict = {"R": pose[:3, :3], "t": pose[:3, 3:] * 1000}

        ply = o3d.read_triangle_mesh(osp.join(DATA_ROOT_PATH, "models/", name, "textured.ply"))
        ply.compute_vertex_normals()
        model = {"pts": np.asarray(ply.vertices) * 1000, "normals": np.asarray(ply.vertex_normals),
                 "colors": np.asarray(ply.vertex_colors), "faces": np.asarray(ply.triangles)}
        r, d = renderer.render(model, (640, 480), intrinsic_matrix, pose_dict["R"], pose_dict["t"])
        a[i, 0].imshow(r)
        a[i, 1].imshow(d)
        ax[0].imshow(np.concatenate([r / 255., 0.7 * np.expand_dims(d != 0, axis=-1)], axis=-1))
        rgb.append(r)
        dep.append(d)
        p_error = pose_error.vsd(pose_dict, gt_pose_dict, model, depth * 1000, 3, 30, intrinsic_matrix)
        if mode == "hull" or mode == "both":
            model_path = DATA_ROOT_PATH + "models"
            pc = linemod_point_cloud(osp.join(model_path, name, "points.xyz"))
            pc_2d, hull = calculate_2d_hull_of_pointcloud(pc, gt_pose[:, :3], gt_pose[:, 3], intrinsic_matrix)
            pred_pc_2d, pred_hull = calculate_2d_hull_of_pointcloud(pc, pose[:3, :3], pose[:3, 3],
                                                                    intrinsic_matrix)
            for simplex in hull.simplices:
                ax[0].plot(pc_2d[simplex, 0], pc_2d[simplex, 1], 'k-')
            for simplex in pred_hull.simplices:
                ax[0].plot(pred_pc_2d[simplex, 0], pred_pc_2d[simplex, 1], 'r-')
        if mode == "vector" or mode == "both":
            p = get_orientation_line_points(gt_pose, intrinsic_matrix)
            icp_p = get_orientation_line_points(pose[:3, :], intrinsic_matrix)
            ax[0].plot([p[0, 0], p[1, 0]], [p[0, 1], p[1, 1]], "r-")
            ax[0].plot([p[0, 0], p[2, 0]], [p[0, 1], p[2, 1]], "g-")
            ax[0].plot([p[0, 0], p[3, 0]], [p[0, 1], p[3, 1]], "y-")
            ax[0].plot([icp_p[0, 0], icp_p[1, 0]], [icp_p[0, 1], icp_p[1, 1]], "r--")
            ax[0].plot([icp_p[0, 0], icp_p[2, 0]], [icp_p[0, 1], icp_p[2, 1]], "g--")
            ax[0].plot([icp_p[0, 0], icp_p[3, 0]], [icp_p[0, 1], icp_p[3, 1]], "y--")
        if show_mask:
            assert masks is not None
            assert item_corr is not None
            j = item_corr[name]
            mask = masks[:, :, j]
            masked = np.ma.masked_where(mask == 0, mask)
            ax[0].imshow(masked, "jet", alpha=0.7)
            ax[1].imshow(masked, "jet", alpha=0.7)
        errors.append(p_error)
    plt.show()
    return errors, rgb, dep

#
# icp_poses, item_corr = get_ransac_RT(r, bboxs, intrinsic_matrix, depth=depth, image=image, voxel_size=0.0001)
# icp_poses, item_corr = get_icp_RT(r, bboxs, intrinsic_matrix)
def order_array_with_respect_to(x, y):
    """

    :param x: array that is to be ordered
    :param y: array that is to be compared to
    :return: ordered indices of x that are present in y
    """
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)
    yindex = np.take(index, sorted_index, mode="clip")
    mask = x[yindex] != y
    return yindex[~mask]

ids = order_array_with_respect_to(r["class_ids"], classes)

errors, rgb, depths = visualize_icp_vs_ground_truth(np.array(image[:, :, :3], dtype=np.int16),
                              depth, r["poses"], r["class_ids"],
                              poses, classes, classes_dict,
                              intrinsic_matrix, r["masks"][ids],
                              show_mask=False, mode="both")
# R = poses[:3, :3, 0]
# t = poses[:3, 3:, 0] * 1000
# mesh = o3d.read_triangle_mesh("/home/christoph/Data/YCB_Video_Dataset/models/002_master_chef_can/textured.ply")
# model = {"pts": np.asarray(mesh.vertices) * 1000, "colors": np.asarray(mesh.vertex_colors), "faces": np.asarray(mesh.triangles)}
# rgb, depth = renderer.render(model, (500, 500), intrinsic_matrix, R, t)