import open3d as o3d
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import copy

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.draw_geometries([source_temp, target])

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = o3d.voxel_down_sample(pcd, voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    o3d.estimate_normals(pcd_down, o3d.KDTreeSearchParamHybrid(
            radius = radius_normal, max_nn = 30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.compute_fpfh_feature(pcd_down,
            o3d.KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size):
    # print(":: Load two point clouds and disturb initial pose.")
    source = o3d.read_point_cloud(scgal_path)
    target = o3d.read_point_cloud(ycbv_path)
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0],
    #                         [1.0, 0.0, 0.0, 0.0],
    #                         [0.0, 1.0, 0.0, 0.0],
    #                         [0.0, 0.0, 0.0, 1.0]])
    # source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            distance_threshold,
            o3d.TransformationEstimationPointToPoint(False), 4,
            [o3d.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            o3d.RANSACConvergenceCriteria(4000000, 500))
    return result

def refine_registration(source, target, voxel_size, result_ransac):
    distance_threshold = voxel_size * 2
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration_icp(source, target, distance_threshold,
            result_ransac.transformation,
            o3d.TransformationEstimationPointToPlane())
    return result

def refine_registration_color(source, target, voxel_size, result_ransac):
    distance_threshold = voxel_size * 2
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration_colored_icp(source, target, distance_threshold,
            result_ransac.transformation,
            o3d.TransformationEstimationPointToPlane())
    return result

if __name__ == '__main__':
    base_dir = "/media/pohl/Hitachi/YCB_Video_Dataset/data/0024"
    color_path = osp.join(base_dir, "000001-color.png")
    depth_path = osp.join(base_dir, "000001-depth.png")
    scgal_path = "/common/homes/staff/pohl/Code/Cpp/simox-cgal/data/objects/ycb/black_and_decker_lithium_drill_driver_unboxed/black_and_decker_lithium_drill_driver_unboxed.ply"
    ycbv_path = "/media/pohl/Hitachi/YCB_Video_Dataset/models/035_power_drill/textured.ply"
    # color_raw = o3d.read_image(color_path)
    # depth_raw = o3d.read_image(depth_path)
    # rgbd_image = o3d.create_rgbd_image_from_color_and_depth(
    #     color_raw, depth_raw, depth_scale=10000, convert_rgb_to_intensity=False)
    # print(rgbd_image)
    # plt.subplot(1, 2, 1)
    # plt.title('Redwood grayscale image')
    # plt.imshow(rgbd_image.color)
    # plt.subplot(1, 2, 2)
    # plt.title('Redwood depth image')
    # plt.imshow(np.asarray(rgbd_image.depth)/ 10000)
    # plt.show()
    # pcd = o3d.create_point_cloud_from_rgbd_image(rgbd_image, o3d.PinholeCameraIntrinsic(
    #     o3d.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # # Flip it, otherwise the pointcloud will be upside down
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # o3d.draw_geometries([pcd])
    # source = o3d.read_point_cloud(scgal_path)
    # target = o3d.read_point_cloud(ycbv_path)

    # # draw initial alignment
    # current_transformation = np.identity(4)
    # draw_registration_result_original_color(
    #     source, target, current_transformation)

    # # point to plane ICP
    # current_transformation = np.identity(4)
    # print("2. Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. Distance threshold 0.02.")
    # result_icp = o3d.registration_icp(source, target, 0.5,
    #                               current_transformation, o3d.TransformationEstimationPointToPlane(),
    #                                           o3d.ICPConvergenceCriteria(relative_fitness=1e-6,
    #                                                                  relative_rmse=1e-6, max_iteration=30))
    # print(result_icp)
    # draw_registration_result_original_color(
    #     source, target, result_icp.transformation)

    # colored pointcloud registration
    # This is implementation of following paper
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017
    # deg = 130
    # rot = np.array(
    #     [[np.cos(np.deg2rad(deg)), -np.sin(np.deg2rad(deg)), 0], [np.sin(np.deg2rad(deg)), np.cos(np.deg2rad(deg)), 0],
    #      [0, 0, 1]])
    # voxel_radius = [0.1, 0.05, 0.01, 0.005]
    # max_iter = [600, 500, 300, 140]
    # current_transformation = np.identity(4)
    # # current_transformation[:3, :3] = rot
    # print("3. Colored point cloud registration")
    # for scale in range(4):
    #     iter = max_iter[scale]
    #     radius = voxel_radius[scale]
    #     print([iter, radius, scale])
    #
    #     print("3-1. Downsample with a voxel size %.2f" % radius)
    #     source_down = o3d.voxel_down_sample(source, radius / 2)
    #     target_down = o3d.voxel_down_sample(target, radius / 2)
    #     # o3d.draw_geometries([source_down, target_down])
    #     print("3-2. Estimate normal.")
    #     o3d.estimate_normals(source_down, o3d.KDTreeSearchParamHybrid(
    #         radius=radius * 4, max_nn=50))
    #     o3d.estimate_normals(target_down, o3d.KDTreeSearchParamHybrid(
    #         radius=radius * 4, max_nn=50))
    #
    #     print("3-3. Applying colored point cloud registration")
    #     result_icp = o3d.registration_colored_icp(source_down, target_down,
    #                                           0.8, current_transformation,
    #                                           o3d.ICPConvergenceCriteria(relative_fitness=1e-10,
    #                                                                  relative_rmse=1e-10, max_iteration=iter), 0.92)
    #     current_transformation = result_icp.transformation
    #     print(result_icp)
    # draw_registration_result_original_color(
    #     source, target, result_icp.transformation)

    voxel_size = 0.005  # means 5cm for the dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
        prepare_dataset(voxel_size)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh, voxel_size)
    print(result_ransac)
    draw_registration_result(source_down, target_down,
                             result_ransac.transformation)

    result_icp = refine_registration(source, target,
                                     source_fpfh, target_fpfh, voxel_size, result_ransac)
    print(result_icp)
    draw_registration_result(source, target, result_icp.transformation)
