from tf_ops.nn_distance.tf_nndistance import nn_distance



import numpy as np
import pickle as pkl
import tensorflow as tf
import os
import keras.backend as K

def create_xyz_dataframe(dataset, path_to_dataset):
    models = os.path.join(path_to_dataset, "models")
    dic = []
    # dix = {}
    for cls in dataset.class_info:
        if cls["name"] == "BG":
            pc = np.zeros((2620, 3))
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
    :param pred_trans: [N, 1, 3]
    :param target_rot: [N, 3, 3]
    :param target_trans: [N, 1, 3]
    :param pos_obj_models: [N, 3, num_points]
    :return:
    """
    # pred_rot_shape = tf.shape(pred_rot)
    # pred_trans_shape = tf.shape(pred_trans)
    # target_rot_shape = tf.shape(target_rot)
    # target_trans_shape = tf.shape(target_trans)
    # pos_obj_models_shape = tf.shape(pos_obj_models)
    # print_op = tf.print([pred_rot_shape, pred_trans_shape, target_rot_shape, target_trans_shape, pos_obj_models_shape])
    # with tf.control_dependencies([print_op]):
    pred_models =tf.transpose(tf.matmul(pred_rot, pos_obj_models), (0, 2, 1),
                              name="transpose_pred_models")
    pred_models = tf.add(pred_models, pred_trans, name="added_pred_models")
    target_models = tf.transpose(tf.matmul(target_rot, pos_obj_models), (0, 2, 1),
                                 name="transposed_target_models")
    target_models = tf.add(target_models, target_trans, name="added_target_models")
    dis1, ind1, dis2, ind2 = nn_distance(pred_models, target_models)
    loss = tf.math.xdivy(tf.reduce_sum(dis1) + tf.reduce_sum(dis2), 2, name="chamfer_loss")
    return loss

def mrcnn_pose_loss_graph(target_poses, target_class_ids, pred_trans, pred_rot, xyz_models):
    """

    :param target_poses: [batch, num_rois, 4, 4] Translation Matrix
    :param target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    :param pred_trans: [batch, num_rois, 3, NUM_CLASSES]
    :param pred_rot: [batch, num_rois, 3, 3, NUM_CLASSES]
    :param xyz_models: [NUM_CLASSES, N, 3] point cloud models of the
                        different classes, where N is the number of points
                        in the model
    :return:
    """
    # pred_trans, pred_rot = pred_poses
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_poses = K.reshape(target_poses, (-1, 4, 4))
    # TODO: check if this has shape [N, 3, 1] or [N, 3]; We need the former
    target_trans = K.reshape(target_poses[:, 3, 0:3], (-1, 1, 3)) # shape: [batch * num_rois, 1, 3]
    target_rot = K.reshape(target_poses[:, 0:3, 0:3], (-1, 3, 3)) # shape: [batch * num_rois, 3, 3]
    num_classes = tf.shape(pred_trans)[0]
    pred_trans = K.reshape(pred_trans, (-1, 1, 3, num_classes)) # shape: [batch * num_rois, 3, num_classes]
    pred_rot = K.reshape(pred_rot, (-1, 3, 3, num_classes)) #  shape: [batch * num_rois, 3, 3, num_classes]
    # Permute predicted tensors to [N, num_classes, ...]
    pred_trans = tf.transpose(pred_trans, [0, 3, 1, 2], name="pred_trans") # shape: [batch * num_rois, num_classes, 1, 3]
    pred_rot = tf.transpose(pred_rot, [0, 3, 1, 2], name="pred_rot") # shape: [batch *  num_rois, num_classes, 3, 3]
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64, name="positive_class_ids")
    indices = tf.stack([positive_ix, positive_class_ids], axis=1, name="indices")

    # Gather the masks (predicted and true) that contribute to loss
    y_true_t = tf.gather(target_trans, positive_ix, name="y_true_t") # shape: [pos_ix, 3]
    y_true_r = tf.gather(target_rot, positive_ix, name="y_true_r")   # shape: [pos_ix, 3, 3]
    y_pred_t = tf.gather_nd(pred_trans, indices, name="y_pred_t")    # shape: [pos_ix, 3]
    y_pred_r = tf.gather_nd(pred_rot, indices, name="y_pred_r")      # shape: [pos_ix, 3, 3]
    pos_xyz_models = tf.gather(xyz_models, positive_class_ids, name="pos_xyz_models") # [pos_ix, N, 3]
    y_true_t_shape = tf.shape(y_true_t) # shape: [pos_ix, 3]
    y_true_r_shape = tf.shape(y_true_r)  # shape: [pos_ix, 3, 3]
    y_pred_t_shape = tf.shape(y_pred_t)  # shape: [pos_ix, 3]
    y_pred_r_shape = tf.shape(y_pred_r)  # shape: [pos_ix, 3, 3]
    pos_xyz_models_shape = tf.shape(pos_xyz_models)  # [pos_ix, N, 3]
    print_op = tf.print([y_pred_r_shape,y_true_r_shape, y_pred_t_shape, y_pred_r_shape, pos_xyz_models_shape])

    # with tf.control_dependencies([print_op]):
    loss = K.switch(tf.size(y_true_r) > 0,
                    chamfer_distance_loss(y_pred_r, y_pred_t, y_true_r, y_true_t, pos_xyz_models),
                    tf.constant(0.0))
    # loss = K.mean(loss)
    return loss

class ChamferLossTest(tf.test.TestCase):

    def testRotationConvergence(self):
        with open("/home/christoph/Code/Python/Mask_RCNN/samples/YCB_Video/XYZ_Models.pkl", "rb") as f:
            # pkl.dump(df, f)
            df = np.array(pkl.load(f), dtype=np.float32)
        pos_obj_models = tf.constant(df)
        pos_trans = tf.transpose(pos_obj_models, (0, 2, 1))
        rot1 = [[1., 0, 0], [0, 1., 0], [0, 0, 1.]]
        rot2 = [[0.8, 0.1, 0], [0.2, 0.8, 0], [0, 0.1, 0.9]]
        trans1 = [[0., 0., 0.]]
        tf_rot1 = tf.constant(np.tile(rot1, (22, 1, 1)), dtype=tf.float32)
        tf_rot2 = tf.Variable(np.tile(rot2, (22, 1, 1)), dtype=tf.float32)
        tf_trans1 = tf.constant(np.tile(trans1, (22, 1, 1)), dtype=tf.float32)
        chamfer = chamfer_distance_loss(tf_rot1, tf_trans1, tf_rot2, tf_trans1, pos_trans)
        train = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(chamfer)
        print_op = tf.print("Rotation: ", tf_rot2)
        counter = 0
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(3000):
                counter += 1
                trainloss,_=sess.run([chamfer,train])
                # if counter == 1000:
                #     sess.run(print_op)
                #     counter = 0
            self.assertAllClose(tf_rot1.eval()[1:], tf_rot2.eval()[1:], rtol=1e-4)

    def testTranslationConvergence(self):
        with open("/home/christoph/Code/Python/Mask_RCNN/samples/YCB_Video/XYZ_Models.pkl", "rb") as f:
            # pkl.dump(df, f)
            df = np.array(pkl.load(f), dtype=np.float32)
        pos_obj_models = tf.constant(df)
        pos_trans = tf.transpose(pos_obj_models, (0, 2, 1))
        rot1 = [[1., 0, 0], [0, 1., 0], [0, 0, 1.]]
        rot2 = [[1., 0, 0], [0, 1., 0], [0, 0, 1.]]
        trans1 = [[0., 0., 0.]]
        trans2 = [[0.2, -0.1, 0.3]]
        tf_rot1 = tf.constant(np.tile(rot1, (22, 1, 1)), dtype=tf.float32)
        tf_rot2 = tf.constant(np.tile(rot2, (22, 1, 1)), dtype=tf.float32)
        tf_trans1 = tf.constant(np.tile(trans1, (22, 1, 1)), dtype=tf.float32)
        tf_trans2 = tf.Variable(np.tile(trans2, (22, 1, 1)), dtype=tf.float32)
        chamfer = chamfer_distance_loss(tf_rot1, tf_trans1, tf_rot2, tf_trans2, pos_trans)
        # TODO: This seems to be extremely volatile and the gradients explode very quickly when learning rate is higher
        train = tf.train.GradientDescentOptimizer(learning_rate=0.00005).minimize(chamfer)
        print_op = tf.print("Translation: ", tf_trans2)
        counter = 0
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(300):
                counter += 1
                trainloss,_=sess.run([chamfer,train])
                # if counter == 10:
                #     sess.run(print_op)
                #     counter = 0
            self.assertAllClose(tf_trans1.eval()[1:], tf_trans2.eval()[1:], rtol=1e-4)

    def testPoseLossGraph(self):
        # TODO: Do it
        return True

if __name__=='__main__':
    from samples.LINEMOD.LINEMOD import linemod_point_cloud
    from samples.YCB_Video.YCB_Video import YCBVDataset

    YCB_path = "/home/christoph/Hitachi/YCB_Video_Dataset/"
    xyz_file = "models/035_power_drill/points.xyz"
    print(f"Optimizing Loss for {xyz_file}")
    dataset = YCBVDataset()
    dataset.load_ycbv(YCB_path, "trainval")
    dataset.prepare()
    image = dataset.load_image(dataset.image_from_source_map["YCBV.0081/000982"])
    mask, classes = dataset.load_mask(dataset.image_from_source_map["YCBV.0081/000982"])
    poses, pclasses = dataset.load_pose(dataset.image_from_source_map["YCBV.0081/000982"])
    # df = create_xyz_dataframe(dataset, YCB_path)
    tf.test.main()

