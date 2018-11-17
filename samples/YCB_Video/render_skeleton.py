import xml.etree.ElementTree as et
import pandas as pd
import scipy.io as io
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import open3d as o3d
import cv2
import os
import subprocess as sp
from shutil import copyfile

from mpl_toolkits.mplot3d import Axes3D
from samples.YCB_Video.open3d_test import draw_registration_result, preprocess_point_cloud, execute_global_registration,\
    refine_registration
# ROOT_DIR = os.path.abspath("../../")
# sys.path.append(ROOT_DIR)

def rot_around_x_axis(points, deg):
    rot = np.array([[1, 0, 0], [0, np.cos(np.deg2rad(deg)), -np.sin(np.deg2rad(deg))],
                    [0, np.sin(np.deg2rad(deg)), np.cos(np.deg2rad(deg))]])
    return np.matmul(rot, points.transpose()).transpose(), rot

def rot_around_y_axis(points, deg):
    rot = np.array([[np.cos(np.deg2rad(deg)), 0, np.sin(np.deg2rad(deg))], [0, 1, 0],
                    [-np.sin(np.deg2rad(deg)), 0, np.cos(np.deg2rad(deg))]])
    return np.matmul(rot, points.transpose()).transpose(), rot

def rot_around_z_axis(points, deg):
    rot = np.array([[np.cos(np.deg2rad(deg)), -np.sin(np.deg2rad(deg)), 0], [ np.sin(np.deg2rad(deg)), np.cos(np.deg2rad(deg)), 0],
                    [0, 0, 1]])
    return np.matmul(rot, points.transpose()).transpose(), rot

def load_YCB_meta_infos(id):
    path = "/media/pohl/Hitachi/YCB_Video_Dataset/data/%s-meta.mat" % id
    meta = io.loadmat(path)
    int_matrix = meta["intrinsic_matrix"]
    classes = meta["cls_indexes"]
    depth_factor = meta["factor_depth"]
    rot_trans_mat = meta["rotation_translation_matrix"]
    vertmap = meta["vertmap"]
    poses = meta["poses"]
    center = meta["center"]
    return int_matrix, classes, depth_factor, rot_trans_mat, vertmap, poses, center

def get_skeleton_points(path):
    tree = et.parse(path)
    root = tree.getroot()
    skel_p = {}
    for point in root.iter("Point"):
        id = point.attrib["index"]
        coord = point.find("Coordinate")
        skel_p[id] = coord.attrib
    return pd.DataFrame(data=skel_p, dtype=np.float32).transpose()


def load_classes_id_dict():
    path = "/media/pohl/Hitachi/YCB_Video_Dataset/image_sets/classes.txt"
    d = {}
    with open(path, "r") as f:
        for i, val in enumerate(f):
            d[i + 1] = val[:-1]
    return d

def get_point_cloud(path):
    "returns a PC in meters"
    pc = []

    def str_to_list_of_floats(line):
        m = map(float, line[:-2].split(" "))
        return list(m)

    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            l = str_to_list_of_floats(line)
            pc.append(l[:3])
    return np.array(pc)

def make_point_cloud_from_mesh_xml(path):
    tree = et.parse(path)
    root = tree.getroot()
    pc_p = {}
    for vertex in root.find("Vertices").iter("Vertex"):
        id = vertex.attrib["index"]
        point = vertex.find("Point")
        pc_p[id] = point.attrib
    return pd.DataFrame(data=pc_p, dtype=np.float32).transpose()

def load_scgal_names_dict(path):
    scgal_names = {}
    with open(path) as f:
        for _, line in enumerate(f):
            ycb_name, scgal_name = line.split()
            scgal_names[ycb_name] = scgal_name
    return scgal_names

def render_skeleton_image(img_id, model_name):
    data_path = "/media/pohl/Hitachi/YCB_Video_Dataset/data/"
    models_path = "/media/pohl/Hitachi/YCB_Video_Dataset/models"
    ycb_scgal_names_file = osp.join(models_path, "cgal_skeletons.txt")
    rgb_file = osp.join(data_path, img_id + "-color.png")
    pc_model_file = osp.join(models_path, model_name, "points.xyz")
    model_folder = osp.join(models_path, model_name)

    classes_dict = load_classes_id_dict()
    scgal_names = load_scgal_names_dict(ycb_scgal_names_file)
    scgal_data_folder = "/common/homes/staff/pohl/Code/Cpp/simox-cgal/data/objects/ycb"
    scgal_model_folder = osp.join(scgal_data_folder, scgal_names[model_name])
    # the naming of these files is absolute bogus
    xml_file_name = scgal_names[model_name]
    if "-" in xml_file_name:
        xml_file_name = xml_file_name.split("-")[1]
    skeleton_file = osp.join(scgal_model_folder, "CGALSkeleton", "skeleton", "CGALMesh-" + xml_file_name + ".xml")
    mesh_file = osp.join(scgal_model_folder, "CGALSkeleton", "mesh", "CGALSkeleton-" + xml_file_name + ".xml")
    try:
        skel_points = get_skeleton_points(skeleton_file).values / 1000
        scgal_pc = make_point_cloud_from_mesh_xml(mesh_file).values / 1000
    except FileNotFoundError:
        skeleton_file = skeleton_file[:-4] + "-poisson.xml"
        mesh_file = mesh_file[:-4] + "-poisson.xml"
        skel_points = get_skeleton_points(skeleton_file).values / 1000
        scgal_pc = make_point_cloud_from_mesh_xml(mesh_file).values / 1000

    model_id = None
    # image = cv2.imread(rgb_file, -1)
    intrinsic_matrix, classes, depth_factor, rot_trans_mat, vertmap, poses, center = load_YCB_meta_infos(img_id)
    for id, value in classes_dict.items():
        if value == model_name:
            model_id = np.where(classes == id)[0][0]
    # model_pc = get_point_cloud(pc_model_file)
    int_points, rot = rot_around_z_axis(scgal_pc, 130)
    z_rot = np.zeros((4, 4))
    z_rot[3, 3] = 1
    z_rot[:3, :3] = rot
    if osp.exists("mask.xyz"):
        os.remove("mask.xyz")
    with open("mask.xyz", "w") as f:
        for row in int_points:
            # f.write("%.5f %.5f %.5f\n"%(row[0] - mean[0], row[1] - mean[1], row[2] - mean[2]))
            f.write("%.5f %.5f %.5f\n" % (row[0], row[1], row[2]))
    sp.run(["pcl_xyz2pcd", "mask.xyz", "mask.pcd"], stdout=sp.DEVNULL, check=True)
    if not osp.exists(osp.join(model_folder, "model_downsampled.pcd")):
        sp.run(["pcl_obj2pcd", osp.join(model_folder, "textured.obj"), osp.join(model_folder, "model.pcd")],
               stdout=sp.DEVNULL, check=True)
        sp.run(["pcl_uniform_sampling", osp.join(model_folder, "model.pcd"),
                osp.join(model_folder, "model_downsampled.pcd"), "-radius", "0.002"], stdout=sp.DEVNULL, check=True)
    # sp.run(["pcl_xyz2pcd", osp.join(model_path, "points.xyz"), "model.pcd"], stdout=sp.DEVNULL, check=True)
    copyfile(osp.join(model_folder, "model_downsampled.pcd"), "model.pcd")
    completed_process = sp.run(["pcl_icp", "-d", "0.1", "-r", "0.1", "model.pcd", "mask.pcd"], stdout=sp.PIPE, check=True)
    str_arr = str(completed_process.stdout).split("\\n")[6:10]
    # sp.run(["pcl_viewer", "model.pcd", "mask.pcd"])
    int_pose = np.array([np.array(row.split(), np.float32) for row in str_arr])
    scgal_to_ycb_pose = np.matmul(int_pose, z_rot)
    p_hom = np.column_stack((skel_points, np.ones(skel_points.shape[0]))).transpose()
    pc_hom = np.column_stack((scgal_pc, np.ones(scgal_pc.shape[0]))).transpose()
    p_obj = np.matmul(scgal_to_ycb_pose, p_hom)
    pc_obj = np.matmul(scgal_to_ycb_pose, pc_hom)
    obj_to_camera_pose = poses[:, :, model_id]
    p_cam = np.matmul(obj_to_camera_pose, p_obj).transpose()
    pc_cam = np.matmul(obj_to_camera_pose, pc_obj).transpose()
    p_2d, _ = cv2.projectPoints(p_cam, np.array([0, 0, 0], np.float32), np.array([0, 0, 0], np.float32),
                                intrinsic_matrix, 0)
    p_2d = p_2d.reshape(-1, 2)
    pc_2d, _ = cv2.projectPoints(pc_cam, np.array([0, 0, 0], np.float32), np.array([0, 0, 0], np.float32),
                                intrinsic_matrix, 0)
    pc_2d = pc_2d.reshape(-1, 2)
    return p_2d, pc_2d

def render_skeleton_image_ransac(img_id, model_name, voxel_size=0.005):
    data_path = "/media/pohl/Hitachi/YCB_Video_Dataset/data/"
    models_path = "/media/pohl/Hitachi/YCB_Video_Dataset/models"
    ycb_scgal_names_file = osp.join(models_path, "cgal_skeletons.txt")
    rgb_file = osp.join(data_path, img_id + "-color.png")
    pc_model_file = osp.join(models_path, model_name, "model.pcd")
    model_folder = osp.join(models_path, model_name)

    classes_dict = load_classes_id_dict()
    scgal_names = load_scgal_names_dict(ycb_scgal_names_file)
    scgal_data_folder = "/common/homes/staff/pohl/Code/Cpp/simox-cgal/data/objects/ycb"
    scgal_model_folder = osp.join(scgal_data_folder, scgal_names[model_name])
    # the naming of these files is absolute bogus
    xml_file_name = scgal_names[model_name]
    if "-" in xml_file_name:
        xml_file_name = xml_file_name.split("-")[1]
    skeleton_file = osp.join(scgal_model_folder, "CGALSkeleton", "skeleton", "CGALMesh-" + xml_file_name + ".xml")
    mesh_file = osp.join(scgal_model_folder, "CGALSkeleton", "mesh", "CGALSkeleton-" + xml_file_name + ".xml")
    try:
        skel_points = get_skeleton_points(skeleton_file).values / 1000
        scgal_xyz = make_point_cloud_from_mesh_xml(mesh_file).values / 1000
    except FileNotFoundError:
        skeleton_file = skeleton_file[:-4] + "-poisson.xml"
        mesh_file = mesh_file[:-4] + "-poisson.xml"
        skel_points = get_skeleton_points(skeleton_file).values / 1000
        scgal_xyz = make_point_cloud_from_mesh_xml(mesh_file).values / 1000
    scgal_pc = o3d.PointCloud()
    scgal_pc.points = o3d.Vector3dVector(scgal_xyz)
    scgal_pcd, _ = preprocess_point_cloud(scgal_pc, voxel_size / 5)
    # o3d.estimate_normals(scgal_pc, o3d.KDTreeSearchParamHybrid(
    #     radius=voxel_size * 2, max_nn=30))

    model_id = None
    # image = cv2.imread(rgb_file, -1)
    intrinsic_matrix, classes, depth_factor, rot_trans_mat, vertmap, poses, center = load_YCB_meta_infos(img_id)
    for id, value in classes_dict.items():
        if value == model_name:
            model_id = np.where(classes == id)[0][0]
    if not osp.exists(pc_model_file):
        sp.run(["pcl_obj2pcd", osp.join(model_folder, "textured.obj"), osp.join(model_folder, "model.pcd")],
               stdout=sp.DEVNULL, check=True)
    ycbv_pc = o3d.read_point_cloud(pc_model_file)
    ycbv_pcd, _ = preprocess_point_cloud(ycbv_pc, voxel_size / 5)
    # o3d.estimate_normals(ycbv_pc, o3d.KDTreeSearchParamHybrid(
    #     radius=voxel_size * 2, max_nn=30))
    ycbv_pcd_down, ycbv_fpfh = preprocess_point_cloud(ycbv_pc, voxel_size)
    scgal_pcd_down, scgal_fpfh = preprocess_point_cloud(scgal_pc, voxel_size)
    result_ransac = execute_global_registration(scgal_pcd_down, ycbv_pcd_down,
                                scgal_fpfh, ycbv_fpfh, voxel_size)
    draw_registration_result(scgal_pcd_down, ycbv_pcd_down,
                             result_ransac.transformation)
    result_icp = refine_registration(scgal_pcd, ycbv_pcd, voxel_size, result_ransac)
    # draw_registration_result(scgal_pc, ycbv_pc, result_icp.transformation)
    # sp.run(["pcl_viewer", "model.pcd", "mask.pcd"])
    scgal_to_ycb_pose = result_icp.transformation
    p_hom = np.column_stack((skel_points, np.ones(skel_points.shape[0]))).transpose()
    pc_hom = np.column_stack((scgal_xyz, np.ones(scgal_xyz.shape[0]))).transpose()
    p_obj = np.matmul(scgal_to_ycb_pose, p_hom)
    pc_obj = np.matmul(scgal_to_ycb_pose, pc_hom)
    obj_to_camera_pose = poses[:, :, model_id]
    p_cam = np.matmul(obj_to_camera_pose, p_obj).transpose()
    pc_cam = np.matmul(obj_to_camera_pose, pc_obj).transpose()
    p_2d, _ = cv2.projectPoints(p_cam, np.array([0, 0, 0], np.float32), np.array([0, 0, 0], np.float32),
                                intrinsic_matrix, 0)
    p_2d = p_2d.reshape(-1, 2)
    pc_2d, _ = cv2.projectPoints(pc_cam, np.array([0, 0, 0], np.float32), np.array([0, 0, 0], np.float32),
                                intrinsic_matrix, 0)
    pc_2d = pc_2d.reshape(-1, 2)
    return p_2d, pc_2d

if __name__ == "__main__":
    skeleton_file = "/common/homes/staff/pohl/Code/Cpp/simox-cgal/data/objects/ycb/black_and_decker_lithium_drill_driver_unboxed/CGALSkeleton/skeleton/CGALMesh-black_and_decker_lithium_drill_driver_unboxed.xml"
    rgb_file = "/media/pohl/Hitachi/YCB_Video_Dataset/data/0024/000001-color.png"
    dpt_file = "/media/pohl/Hitachi/YCB_Video_Dataset/data/0024/000001-depth.png"
    pc_file = "/media/pohl/Hitachi/YCB_Video_Dataset/models/035_power_drill/points.xyz"
    mesh_file = "/common/homes/staff/pohl/Code/Cpp/simox-cgal/data/objects/ycb/black_and_decker_lithium_drill_driver_unboxed/CGALSkeleton/mesh/CGALSkeleton-black_and_decker_lithium_drill_driver_unboxed.xml"
    model_path = osp.join("/media/pohl/Hitachi/YCB_Video_Dataset/models", "035_power_drill")

    points = get_skeleton_points(skeleton_file)
    image = cv2.imread(rgb_file, -1)
    depth = cv2.imread(dpt_file, -1)
    classes_dict = load_classes_id_dict()
    intrinsic_matrix, classes, depth_factor, rot_trans_mat, vertmap, poses, center = load_YCB_meta_infos("0024/000001")

    pc = get_point_cloud(pc_file)
    #
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(image)
    # ax[1].imshow(depth)
    # # p_1 = rot_around_y_axis(points.values / 1000, 180)
    # # p_2 = rot_around_y_axis(p_1, 150)
    # # p_2[:, 1] -= .5 * (np.max(pc[:, 1]) - np.min(pc[:, 1]))
    #
    # # p_hom = np.column_stack((p_2, np.ones(p_2.shape[0]))).transpose()
    #
    #
    #
    # scgal_pc = make_point_cloud_from_mesh_xml(mesh_file).values
    # p_1, rot = rot_around_z_axis(scgal_pc / 1000, 130)
    # rot_1 = np.zeros((4, 4))
    # rot_1[3, 3] = 1
    # rot_1[:3, :3] = rot
    #
    #
    # if osp.exists("mask.xyz"):
    #     os.remove("mask.xyz")
    # with open("mask.xyz", "w") as f:
    #     for row in p_1:
    #         # f.write("%.5f %.5f %.5f\n"%(row[0] - mean[0], row[1] - mean[1], row[2] - mean[2]))
    #         f.write("%.5f %.5f %.5f\n" % (row[0], row[1], row[2]))
    # sp.run(["pcl_xyz2pcd", "mask.xyz", "mask.pcd"], stdout=sp.DEVNULL, check=True)
    # if not osp.exists(osp.join(model_path, "model_downsampled.pcd")):
    #     sp.run(["pcl_obj2pcd", osp.join(model_path, "textured.obj"), osp.join(model_path, "model.pcd")],
    #            stdout=sp.DEVNULL, check=True)
    #     sp.run(["pcl_uniform_sampling", osp.join(model_path, "model.pcd"),
    #             osp.join(model_path, "model_downsampled.pcd"), "-radius", "0.002"], stdout=sp.DEVNULL, check=True)
    # # sp.run(["pcl_xyz2pcd", osp.join(model_path, "points.xyz"), "model.pcd"], stdout=sp.DEVNULL, check=True)
    # copyfile(osp.join(model_path, "model_downsampled.pcd"), "model.pcd")
    # completed_process = sp.run(["pcl_icp", "-d", "0.1", "-r", "0.1", "model.pcd", "mask.pcd"], stdout=sp.PIPE, check=True)
    # str_arr = str(completed_process.stdout).split("\\n")[6:10]
    # # sp.run(["pcl_viewer", "model.pcd", "mask.pcd"])
    # pose = np.array([np.array(row.split(), np.float32) for row in str_arr])
    # result_rot = np.matmul(pose, rot_1)
    # p_hom = np.column_stack((points.values / 1000, np.ones(points.values.shape[0]))).transpose()
    # p = np.matmul(result_rot, p_hom)
    # pose = poses[:, :, 4]
    # p_cam = np.matmul(pose, p).transpose()
    # p_2d, _ = cv2.projectPoints(p_cam, np.array([0, 0, 0], np.float32), np.array([0, 0, 0], np.float32),
    #                             intrinsic_matrix, 0)
    # p_2d = p_2d.reshape(-1, 2)
    # ax[0].scatter(p_2d[:, 0], p_2d[:, 1])
    # fig.tight_layout()
    # plt.show()
    # # completed_process = sp.run(["pcl_icp", "-d", "0.01", "-r", "0.01", "-i", "100", "model.pcd", "mask.pcd"],
    # #                            stdout=sp.PIPE, check=True)
    # # str_arr = str(completed_process.stdout).split("\\n")[6:10]
    # # sp.run(["pcl_viewer", "model.pcd", "mask.pcd"])
    # # pose2 = np.array([np.array(row.split(), np.float32) for row in str_arr])
    # # # pose[:3, :3] = np.matmul(pose2[:3, :3], pose[:3, :3])
    # # # pose[:3, 3] += pose2[:3, 3]
    # # poses[key] = np.matmul(pose2, pose)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(p_2[:, 0], # - .5 * (np.max(pc[:, 0]) - np.min(pc[:, 0])),
    # #            p_2[:, 1], p_2[:, 2])
    # ax.scatter(p_1[:, 0], p_1[:, 1], p_1[:, 2])
    # ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2])
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    fig = plt.figure()
    ax = fig.add_subplot("111")
    ax.imshow(image)
    # for id in classes:
    #     print(id)
    p_2d, pc_2d = render_skeleton_image_ransac("0024/000001", classes_dict[15])
    ax.scatter(p_2d[:, 0], p_2d[:, 1])
    ax.scatter(pc_2d[:, 0], pc_2d[:, 1])


