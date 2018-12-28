from tf_ops.nn_distance.tf_nndistance import nn_distance
from samples.LINEMOD.LINEMOD import linemod_point_cloud
import pandas as pd
import os

def create_xyz_dataframe(path_to_dataset):
    models = os.path.join(path_to_dataset, "models")
    data = []
    for fol in os.listdir(models):
        if not os.path.isdir(os.path.join(models, fol)):
            continue
        path = os.path.join(models, fol, "points.xyz")
        pc = linemod_point_cloud(path)
        dic[fol] = pc
    return pd.DataFrame(dic)

def chamfer_distance_loss(pred_rot, pred_trans, target_rot, target_trans, positive_class_ids):
    """
    Calculates the chamfer distance of the predicted rotated pointcloud to the true rotated pointcloud
    for each prediction
    :param pred_rot: [N, 3, 3]
    :param pred_trans: [N, 3]
    :param target_rot: [N, 3, 3]
    :param target_trans: [N, 3]
    :param positive_class_ids: [N]
    :return:
    """


if __name__=='__main__':
    YCB_path = "/home/christoph/Hitachi/YCB_Video_Dataset/"
    xyz_file = "models/035_power_drill/points.xyz"
    print(f"Optimizing Loss for {xyz_file}")
    df = create_xyz_dataframe(YCB_path)