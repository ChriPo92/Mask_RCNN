from tf_ops.nn_distance.tf_nndistance import nn_distance
from samples.LINEMOD.LINEMOD import linemod_point_cloud
from samples.YCB_Video.YCB_Video import YCBVDataset
import pandas as pd
import numpy as np
import pickle as pkl
import os

def create_xyz_dataframe(dataset, path_to_dataset):
    models = os.path.join(path_to_dataset, "models")
    dic = []
    # dix = {}
    for cls in dataset.class_info:
        if cls["name"] == "BG":
            pc = np.array([[0, 0, 0]])
            # dix[cls["name"]] = pc
            dic.append(pc)
        else:
            for fol in os.listdir(models):
                if cls["name"] == fol[4:]:
                    path = os.path.join(models, fol, "points.xyz")
                    pc = linemod_point_cloud(path)
                    # dix[cls["name"]] = pc
                    dic.append(pc)
                else:
                    continue
    return dic

def chamfer_distance_loss(pred_rot, pred_trans, target_rot, target_trans, pos_obj_models):
    """
    Calculates the chamfer distance of the predicted rotated pointcloud to the true rotated pointcloud
    for each prediction
    :param pred_rot: [N, 3, 3]
    :param pred_trans: [N, 3]
    :param target_rot: [N, 3, 3]
    :param target_trans: [N, 3]
    :param pos_obj_models: [N, num_points, 3]
    :return:
    """


if __name__=='__main__':
    YCB_path = "/home/christoph/Hitachi/YCB_Video_Dataset/"
    xyz_file = "models/035_power_drill/points.xyz"
    print(f"Optimizing Loss for {xyz_file}")
    dataset = YCBVDataset()
    dataset.load_ycbv(YCB_path, "trainval")
    dataset.prepare()
    image = dataset.load_image(dataset.image_from_source_map["YCBV.0081/000982"])
    mask, classes = dataset.load_mask(dataset.image_from_source_map["YCBV.0081/000982"])
    poses, pclasses = dataset.load_pose(dataset.image_from_source_map["YCBV.0081/000982"])
    df = create_xyz_dataframe(dataset, YCB_path)
    with open("/home/christoph/Code/Python/Mask_RCNN/samples/YCB_Video/XYZ_Models.pkl", "wb") as f:
        pkl.dump(df, f)
