import open3d as o3d
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np

if __name__ == '__main__':
    base_dir = "/home/christoph/Hitachi/YCB_Video_Dataset/data/0024"
    color_path = osp.join(base_dir, "000001-color.png")
    depth_path = osp.join(base_dir, "000001-depth.png")
    color_raw = o3d.read_image(color_path)
    depth_raw = o3d.read_image(depth_path)
    rgbd_image = o3d.create_rgbd_image_from_color_and_depth(
        color_raw, depth_raw, depth_scale=10000, convert_rgb_to_intensity=False)
    print(rgbd_image)
    plt.subplot(1, 2, 1)
    plt.title('Redwood grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Redwood depth image')
    plt.imshow(np.asarray(rgbd_image.depth)/ 10000)
    plt.show()
    pcd = o3d.create_point_cloud_from_rgbd_image(rgbd_image, o3d.PinholeCameraIntrinsic(
        o3d.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.draw_geometries([pcd])