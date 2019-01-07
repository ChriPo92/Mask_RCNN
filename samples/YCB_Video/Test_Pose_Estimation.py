from samples.YCB_Video.YCB_Video import YCBVDataset, YCBVConfig
from mrcnn.model import load_image_gt
from scipy.spatial import ConvexHull
import scipy.io as io
# from samples.demo import calculate_2d_hull_of_pointcloud, load_YCB_meta_infos

import tensorflow as tf
import numpy as np
import os.path as osp
import pickle as pkl
import cv2
import matplotlib.pyplot as plt


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
    home = osp.expanduser("~")
    path = osp.join(home, "Hitachi/YCB_Video_Dataset/data/%s-meta.mat" % id)
    meta = io.loadmat(path)
    int_matrix = meta["intrinsic_matrix"]
    classes = meta["cls_indexes"]
    depth_factor = meta["factor_depth"]
    rot_trans_mat = meta["rotation_translation_matrix"]
    vertmap = meta["vertmap"]
    poses = meta["poses"]
    center = meta["center"]
    return int_matrix, classes, depth_factor, rot_trans_mat, vertmap, poses, center



class ImageGTTest(tf.test.TestCase):

    def setUp(self):
        home = osp.expanduser("~")

        self.config = YCBVConfig()
        self.config.USE_DEPTH_AWARE_OPS = True
        self.config.MEAN_PIXEL = np.append(self.config.MEAN_PIXEL, 0.0)
        self.config.IMAGE_CHANNEL_COUNT = 4
        self.dataset = YCBVDataset()
        self.dataset.load_ycbv(osp.join(home, "Hitachi/YCB_Video_Dataset/"),
                               "trainval", use_rgbd=self.config.USE_DEPTH_AWARE_OPS)
        self.dataset.prepare()
        with open(osp.join(home, "Code/Python/Mask_RCNN/samples/YCB_Video/XYZ_Models.pkl"), "rb") as f:
            self.point_clouds = np.array(pkl.load(f), dtype=np.float32)


    def testRotatedObjAndMaskAreAlmostEqual(self):
        np.random.seed(1)
        rand_id = np.random.randint(0, len(self.dataset.image_ids))
        image_id = self.dataset.image_info[rand_id]["id"]
        image, image_meta, class_ids, bbox, mask, pose = load_image_gt(self.dataset,
                                                                       self.config,
                                                                       rand_id)
        intrinsic_matrix, classes, depth_factor, rot_trans_mat, vertmap, poses, center = load_YCB_meta_infos(image_id)
        sort_ids = np.argsort(classes.reshape(-1))
        poses = poses[:, :, sort_ids]
        self.assertAllEqual(pose[:3], poses, "Poses have to match")
        self.assertAllEqual(class_ids, classes.reshape(-1)[sort_ids], "class IDs should be the same")
        for i, cl_id in enumerate(class_ids):
            rot = pose[:3, :3, i]
            trans = pose[:3, 3, i]
            pc_2d, hull_2d = calculate_2d_hull_of_pointcloud(self.point_clouds[cl_id], rot, trans, intrinsic_matrix)
            fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
            ax = fig.add_subplot("111")
            fig.patch.set_facecolor((0, 0, 0))
            # fig.patch.set_visible(False)
            ax.axis('off')
            ax.scatter(pc_2d[:, 0], pc_2d[:, 1], c=(1, 0, 0), antialiased=False)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                hspace=0, wspace=0)
            plt.margins(0, 0)
            # cv2 origin is in the upper left corner
            ax.set_ylim((480, 0))
            ax.set_xlim((0, 640))
            fig.tight_layout(pad=0)
            fig.canvas.draw()
            # save rendered image to buffer and return the data
            data = np.array(fig.canvas.renderer._renderer)[:, :, 0]
            data_mask = data != 0
            plt.close("all")
            self.assertAllClose(mask[:, :, i], data_mask)


        self.assertTrue(False)

if __name__=='__main__':
    tf.test.main()