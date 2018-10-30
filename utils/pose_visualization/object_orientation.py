import numpy as np
import cv2
from scipy.spatial import ConvexHull
def get_orientation_line_points(rot, trans, camera_calibration_matrix, scale=0.05):
    """
    calculates the start and end points of a scaled vector along the unit axis in the object frame in image coordinates
    :return:
    """
    # coordinates in homogeneous coordinates in the object frame
    origin = np.array([0, 0, 0, 1])
    x_axis = np.array([scale, 0, 0, 1])
    y_axis = np.array([0, scale, 0, 1])
    z_axis = np.array([0, 0, scale, 1])
    T_o_c = np.zeros((4, 4))
    T_o_c[:3, :3] = np.array(rot).reshape(3, 3)
    T_o_c[:3, 3] = np.array(trans).reshape(3,)
    T_o_c[3, 3] = 1
    origin_cam = np.matmul(T_o_c, origin)
    x_axis_cam = np.matmul(T_o_c, x_axis)
    y_axis_cam = np.matmul(T_o_c, y_axis)
    z_axis_cam = np.matmul(T_o_c, z_axis)
    coords = np.concatenate((origin_cam[:3], x_axis_cam[:3], y_axis_cam[:3], z_axis_cam[:3])).reshape(-1, 3)
    rvec = np.array([0, 0, 0], np.float32)
    tvec = rvec.copy()
    image_points, _ = cv2.projectPoints(coords, rvec, tvec, camera_calibration_matrix, 0)
    return image_points.reshape(-1, 2)

def calculate_2d_hull_of_pointcloud(pc, rot, trans, camera_calibration_matrix):
    T_o_c = np.zeros((4, 4))
    T_o_c[:3, :3] = np.array(rot).reshape(3, 3)
    T_o_c[:3, 3] = np.array(trans).reshape(3,)
    T_o_c[3, 3] = 1
    hom_pc = cv2.convertPointsToHomogeneous(pc).reshape(-1, 4).transpose()
    pc_cam_hom = np.matmul(T_o_c, hom_pc).transpose().reshape(-1, 1, 4)
    pc_cam = cv2.convertPointsFromHomogeneous(pc_cam_hom).reshape(-1, 3)
    pc_2d, _ = cv2.projectPoints(pc_cam, np.array([0, 0, 0], np.float32), np.array([0, 0, 0], np.float32), camera_calibration_matrix, 0)
    pc_2d = pc_2d.reshape(-1, 2)
    hull = ConvexHull(pc_2d)
    return pc_2d, hull