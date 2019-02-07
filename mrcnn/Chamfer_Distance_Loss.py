from tf_ops.nn_distance.tf_nndistance import nn_distance, NNDistance

import numpy as np
import pickle as pkl
import tensorflow as tf
import os
import keras.backend as K
import keras.layers as KL
import keras.engine as KE



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

class SVD(KL.Layer):
    def __init__(self, **kwargs):
        super(SVD, self).__init__(**kwargs)

    def call(self, input, **kwargs):
        """
        Computes the SVD of a tensor
        input: xyz1: (batch_size,#points_1,3)  the first point cloud
        input: xyz2: (batch_size,#points_2,3)  the second point cloud

        """
        outputs = tf.linalg.svd(input)
        names = ["s", "u", "v"]
        result = [tf.identity(o, name=n) for o, n in zip(list(outputs), names)]
        return result

    def compute_output_shape(self, input_shape):
        """
               output: s: (batch_size,?)   distance from first to second
               output: u:  (batch_size,?, ?)   nearest neighbor from first to second
               output: v: (batch_size,?, ?)   distance from second to first

        """
        return [input_shape[:-1], input_shape, input_shape]


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
    pred_models = tf.transpose(tf.matmul(pred_rot, pos_obj_models), (0, 2, 1),
                               name="transpose_pred_models")
    pred_models = tf.add(pred_models, pred_trans, name="added_pred_models")
    total_number_of_points = tf.shape(pos_obj_models)[0] * tf.shape(pos_obj_models)[2]
    target_models = tf.transpose(tf.matmul(target_rot, pos_obj_models), (0, 2, 1),
                                 name="transposed_target_models")
    target_models = tf.add(target_models, target_trans, name="added_target_models")
    dis1, ind1, dis2, ind2 = nn_distance(pred_models, target_models)
    loss = tf.math.xdivy(tf.reduce_sum(dis1) + tf.reduce_sum(dis2),
                         tf.cast(2 * total_number_of_points, "float32"), name="chamfer_loss")
    return loss

def chamfer_distance_loss_keras(pred_rot, pred_trans, target_rot, target_trans, pos_obj_models):
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
    pred_models = KL.Lambda(lambda y: tf.transpose(tf.matmul(y[0], y[1]), (0, 2, 1)),
                            name="transposed_pred_models")([pred_rot, pos_obj_models])
    total_number_of_points = KL.Lambda(lambda y: tf.shape(y)[0] * tf.shape(y)[2],
                                       name="total_number_of_points")(pos_obj_models)
    pred_models = KL.Add(name="added_pred_models")([pred_models, pred_trans])
    target_models = KL.Lambda(lambda y: tf.transpose(tf.matmul(y[0], y[1]), (0, 2, 1)),
                              name="transposed_target_models")([target_rot, pos_obj_models])
    target_models = KL.Add(name="added_target_models")([target_models, target_trans])
    dis1, ind1, dis2, ind2 = NNDistance(name="NNDistance")([pred_models, target_models])
    reduced_mean1 = KL.Lambda(lambda y: tf.reduce_mean(y),
                              name="reduced_mean1")(dis1)
    reduced_mean2 = KL.Lambda(lambda y: tf.reduce_mean(y),
                              name="reduced_mean2")(dis2)
    added_sum = KL.Lambda(lambda y: y[0] + y[1],
                          name="added_reduced_mean")([reduced_mean1, reduced_mean2])
    # loss = KL.Lambda(lambda y: tf.math.xdivy(y[0], tf.cast(2 * y[1], "float32")),
    #                  name="mrcnn_chamfer_loss")([added_sum, total_number_of_points])
    loss = KL.Lambda(lambda y: tf.math.sqrt(y), name="mrcnn_chamfer_loss")(added_sum)
    return loss


def mrcnn_pose_loss_graph_tf(target_poses, target_class_ids, pred_trans, pred_rot, xyz_models):
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
    target_class_ids = tf.reshape(target_class_ids, (-1,), name="target_class_ids")
    target_poses = tf.reshape(target_poses, (-1, 4, 4), name="target_poses")
    # TODO: check if this has shape [N, 3, 1] or [N, 3]; We need the former
    target_trans = tf.reshape(target_poses[:, 0:3, 3], (-1, 1, 3),
                              name="target_trans")  # shape: [batch * num_rois, 1, 3]
    target_rot = tf.reshape(target_poses[:, 0:3, 0:3], (-1, 3, 3),
                            name="target_trans")  # shape: [batch * num_rois, 3, 3]
    num_classes = tf.shape(pred_trans, name="num_classes")[0]
    pred_trans = tf.reshape(pred_trans, (-1, 1, 3, num_classes),
                            name="pred_trans")  # shape: [batch * num_rois, 1, 3, num_classes]
    pred_rot = tf.reshape(pred_rot, (-1, 3, 3, num_classes),
                          name="pred_rot")  # shape: [batch * num_rois, 3, 3, num_classes]
    # Permute predicted tensors to [N, num_classes, ...]
    pred_trans = tf.transpose(pred_trans, [0, 3, 1, 2],
                              name="pred_trans")  # shape: [batch * num_rois, num_classes, 1, 3]
    pred_rot = tf.transpose(pred_rot, [0, 3, 1, 2], name="pred_rot")  # shape: [batch *  num_rois, num_classes, 3, 3]
    positive_ix = tf.where(target_class_ids > 0, name="positive_ix")[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64, name="positive_class_ids")
    indices = tf.stack([positive_ix, positive_class_ids], axis=1, name="indices")

    # Gather the masks (predicted and true) that contribute to loss
    y_true_t = tf.gather(target_trans, positive_ix, name="y_true_t")  # shape: [pos_ix, 3]
    y_true_r = tf.gather(target_rot, positive_ix, name="y_true_r")  # shape: [pos_ix, 3, 3]
    y_pred_t = tf.gather_nd(pred_trans, indices, name="y_pred_t")  # shape: [pos_ix, 3]
    y_pred_r = tf.gather_nd(pred_rot, indices, name="y_pred_r")  # shape: [pos_ix, 3, 3]
    pos_xyz_models = tf.gather(xyz_models, positive_class_ids, name="pos_xyz_models")  # [pos_ix, N, 3]
    # y_true_t_shape = tf.shape(y_true_t) # shape: [pos_ix, 3]
    # y_true_r_shape = tf.shape(y_true_r)  # shape: [pos_ix, 3, 3]
    # y_pred_t_shape = tf.shape(y_pred_t)  # shape: [pos_ix, 3]
    # y_pred_r_shape = tf.shape(y_pred_r)  # shape: [pos_ix, 3, 3]
    # pos_xyz_models_shape = tf.shape(pos_xyz_models)  # [pos_ix, N, 3]
    # print_op = tf.print([y_pred_r_shape,y_true_r_shape, y_pred_t_shape, y_pred_r_shape, pos_xyz_models_shape])

    # with tf.control_dependencies([print_op]):
    loss = tf.cond(tf.greater(tf.size(y_true_r), tf.constant(0)),
                   lambda: chamfer_distance_loss(y_pred_r, y_pred_t, y_true_r, y_true_t, pos_xyz_models),
                   lambda: tf.constant(0.0), name="loss")
    # loss = K.mean(loss)
    return loss


def mrcnn_pose_loss_graph_keras(target_poses, target_class_ids, pred_trans, pred_rot, xyz_models, config, N):
    """

    :param target_poses: [batch, num_rois, 4, 4] Translation Matrix
    :param target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    :param pred_trans: [batch, num_rois, 3, 1, NUM_CLASSES]
    :param pred_rot: [batch, num_rois, 3, 3, NUM_CLASSES]
    :param xyz_models: [NUM_CLASSES, 3, N] point cloud models of the
                        different classes, where N is the number of points
                        in the model
    :return:
    """
    target_class_ids = KL.Lambda(lambda y: tf.reshape(y, (-1,)),
                                 name="mrcnn_pose_loss/target_class_ids")(
                                 target_class_ids)
    target_poses = KL.Lambda(lambda y: tf.reshape(y, (-1, 4, 4)),
                             name="mrcnn_pose_loss/target_poses")(target_poses)
    # TODO: check if this has shape [N, 3, 1] or [N, 3]; We need the former
    target_trans = KL.Lambda(lambda y: tf.reshape(y[:, :3, 3], (-1, 1, 3)),
                             name="mrcnn_pose_loss/target_trans")(
        target_poses)  # shape: [batch * num_rois, 1, 3]
    target_rot = KL.Lambda(lambda y: tf.reshape(y[:, 0:3, 0:3], (-1, 3, 3)),
                           name="mrcnn_pose_loss/target_rot")(
        target_poses)  # shape: [batch * num_rois, 3, 3]
    # num_classes = tf.shape(pred_trans, name="num_classes")[0]
    pred_trans = KL.Lambda(lambda y: tf.reshape(y, (-1, 1, 3, config.NUM_CLASSES)),
                           name="mrcnn_pose_loss/pred_trans_reshape")(
        pred_trans)  # shape: [batch * num_rois, 1, 3, num_classes]
    pred_rot = KL.Lambda(lambda y: tf.reshape(y, (-1, 3, 3, config.NUM_CLASSES)),
                         name="mrcnn_pose_loss/pred_rot_reshape")(
        pred_rot)  # shape: [batch * num_rois, 3, 3, num_classes]
    # Permute predicted tensors to [N, num_classes, ...]
    pred_trans = KL.Lambda(lambda y: tf.transpose(y, [0, 3, 1, 2]),
                           name="mrcnn_pose_loss/pred_trans_transposed")(
        pred_trans)  # shape: [batch * num_rois, num_classes, 1, 3]
    pred_rot = KL.Lambda(lambda y: tf.transpose(y, [0, 3, 1, 2]),
                         name="mrcnn_pose_loss/pred_rot_transposed")(
        pred_rot)  # shape: [batch *  num_rois, num_classes, 3, 3]
    positive_ix = KL.Lambda(lambda y: tf.where(y > 0)[:, 0],
                            name="mrcnn_pose_loss/positive_ix")(target_class_ids) # shape: [pos_ix]
    positive_class_ids = KL.Lambda(lambda y: tf.cast(
                                   tf.gather(y[0], y[1]), tf.int64),
                                   name="mrcnn_pose_loss/positive_class_ids", output_shape=tuple())(
                                   [target_class_ids, positive_ix]) # shape: [pos_ix]
    indices = KL.Lambda(lambda y: tf.stack([y[0], y[1]], axis=1),
                        name="mrcnn_pose_loss/indices")(
                        [positive_ix, positive_class_ids]) # shape: [pos_ix, 2]

    # Gather the masks (predicted and true) that contribute to loss
    y_true_t = KL.Lambda(lambda y: tf.gather(y[0], y[1]),
                         name="mrcnn_pose_loss/y_true_t", output_shape=(3,))(
                         [target_trans, positive_ix])  # shape: [pos_ix, 1, 3]
    y_true_r = KL.Lambda(lambda y: tf.gather(y[0], y[1]),
                         name="mrcnn_pose_loss/y_true_r", output_shape=(3, 3))(
                         [target_rot, positive_ix])  # shape: [pos_ix, 3, 3]
    y_pred_t = KL.Lambda(lambda y: tf.gather_nd(y[0], y[1]),
                         name="mrcnn_pose_loss/y_pred_t", output_shape=(3,))(
                         [pred_trans, indices])  # shape: [pos_ix, 1, 3]
    y_pred_r = KL.Lambda(lambda y: tf.gather_nd(y[0], y[1]),
                         name="mrcnn_pose_loss/y_pred_r", output_shape=(3, 3,))(
                         [pred_rot, indices])  # shape: [pos_ix, 3, 3]
    # the predicted rotations are not orthogonal, which is needed for rotations
    # to get an orthoganl matrix from any 3x3 matrix we use an SVD where
    # a = U * diag(S) * V^h
    s, u, v = SVD(name="mrcnn_pose_loss/pred_rot_svd")(y_pred_r)
    y_pred_r = KL.Lambda(lambda y: tf.linalg.matmul(y[0], y[1]),
                         name="mrcnn_pose_loss/pred_rot_svd_matmul")([u, v])
    pos_xyz_models = KL.Lambda(lambda y: tf.gather(y[0], y[1]),
                               name="mrcnn_pose_loss/pos_xyz_models",
                               output_shape=(3, N,))(
                               [xyz_models, positive_class_ids])  # [pos_ix, 3, N]

    chamfer_loss = chamfer_distance_loss_keras(y_pred_r, y_pred_t, y_true_r, y_true_t, pos_xyz_models)
    # rot_loss = KL.Lambda(lambda y: tf.reduce_mean(tf.keras.losses.mae(y[0], y[1])),
    #                         name="mrcnn_pose_loss/rot_error")([y_true_r, y_pred_r])
    rot_loss = KL.Lambda(lambda y: tf.reduce_mean(tf.losses.huber_loss(y[0], y[1], delta=2.0)),
                            name="mrcnn_pose_loss/rot_error")([y_true_r, y_pred_r])
    huber_trans_loss = KL.Lambda(lambda y: tf.reduce_mean(huber_loss(tf.norm(y[0]-y[1], axis=-1), 2.0)),
                                name="mrcnn_pose_loss/huber_trans")([y_true_t, y_pred_t])
    # loss = KL.Lambda(lambda y: tf.cond(tf.greater(tf.size(y[2]), tf.constant(0)),
    #                                    lambda: chamfer_distance_loss(y[0], y[1], y[2], y[3], y[4]),
    #                                    lambda: tf.constant(0.0)), name="mrcnn_pose_loss/loss")(
    #     [y_pred_r, y_pred_t, y_true_r, y_true_t, pos_xyz_models])
    # loss = K.switch(tf.greater(tf.size(y_true_r), tf.constant(0)),
    #                 chamfer_distance_loss_keras(y_pred_r, y_pred_t, y_true_r, y_true_t, pos_xyz_models),
    #                 K.constant(0.0))
    # loss = K.mean(loss)
    # It is not possible to add constants (shape ()) with a KL.Add; therefore use a lambda layer
    total_loss = KL.Lambda(lambda y: y[0] + y[1],
                           name="mrcnn_pose_loss/total_loss")([huber_trans_loss, rot_loss])
    return total_loss


def mrcnn_pose_loss_model(classes=22, rois=200, N=2640):
    """

    :param target_poses: [batch, num_rois, 4, 4] Translation Matrix
    :param target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    :param pred_trans: [batch, num_rois, 3, NUM_CLASSES]
    :param pred_rot: [batch, num_rois, 3, 3, NUM_CLASSES]
    :param xyz_models: [NUM_CLASSES, 3, N] point cloud models of the
                        different classes, where N is the number of points
                        in the model
    :return:
    """
    # pred_trans, pred_rot = pred_poses
    input_poses = KL.Input((rois, 4, 4))
    input_class_ids = KL.Input((rois,), dtype="int64")
    input_trans = KL.Input((rois, 3, classes))
    input_rot = KL.Input((rois, 3, 3, classes))
    input_models = KL.Input(batch_shape=(classes, 3, N))
    target_class_ids = KL.Lambda(lambda y: tf.reshape(y, (-1,)), name="target_class_ids")(input_class_ids)
    target_poses = KL.Lambda(lambda y: tf.reshape(y, (-1, 4, 4)), name="target_poses")(input_poses)
    # TODO: check if this has shape [N, 3, 1] or [N, 3]; We need the former
    target_trans = KL.Lambda(lambda y: tf.reshape(y[:, 3, 0:3], (-1, 1, 3)), name="target_trans")(
        target_poses)  # shape: [batch * num_rois, 1, 3]
    target_rot = KL.Lambda(lambda y: tf.reshape(y[:, 0:3, 0:3], (-1, 3, 3)), name="target_rot")(
        target_poses)  # shape: [batch * num_rois, 3, 3]
    # num_classes = tf.shape(pred_trans, name="num_classes")[0]
    pred_trans = KL.Lambda(lambda y: tf.reshape(y, (-1, 1, 3, classes)), name="pred_trans_reshape")(
        input_trans)  # shape: [batch * num_rois, 1, 3, num_classes]
    pred_rot = KL.Lambda(lambda y: tf.reshape(y, (-1, 3, 3, classes)), name="pred_rot_reshape")(
        input_rot)  # shape: [batch * num_rois, 3, 3, num_classes]
    # Permute predicted tensors to [N, num_classes, ...]
    pred_trans = KL.Lambda(lambda y: tf.transpose(y, [0, 3, 1, 2]), name="pred_trans_transposed")(
        pred_trans)  # shape: [batch * num_rois, num_classes, 1, 3]
    pred_rot = KL.Lambda(lambda y: tf.transpose(y, [0, 3, 1, 2]), name="pred_rot_transposed")(
        pred_rot)  # shape: [batch *  num_rois, num_classes, 3, 3]
    positive_ix = KL.Lambda(lambda y: tf.where(y > 0, name="positive_ix")[:, 0])(target_class_ids)
    positive_class_ids = KL.Lambda(lambda y: tf.cast(
        tf.gather(y[0], y[1]), tf.int64), name="positive_class_ids", output_shape=tuple())(
        [target_class_ids, positive_ix])
    indices = KL.Lambda(lambda y: tf.stack([y[0], y[1]], axis=1), name="indices")([positive_ix, positive_class_ids])

    # Gather the masks (predicted and true) that contribute to loss
    y_true_t = KL.Lambda(lambda y: tf.gather(y[0], y[1]), name="y_true_t", output_shape=(3,))(
        [target_trans, positive_ix])  # shape: [pos_ix, 3]
    y_true_r = KL.Lambda(lambda y: tf.gather(y[0], y[1]), name="y_true_r", output_shape=(3, 3))(
        [target_rot, positive_ix])  # shape: [pos_ix, 3, 3]
    y_pred_t = KL.Lambda(lambda y: tf.gather_nd(y[0], y[1]), name="y_pred_t", output_shape=(3,))(
        [pred_trans, indices])  # shape: [pos_ix, 3]
    y_pred_r = KL.Lambda(lambda y: tf.gather_nd(y[0], y[1]), name="y_pred_r", output_shape=(3, 3,))(
        [pred_rot, indices])  # shape: [pos_ix, 3, 3]
    pos_xyz_models = KL.Lambda(lambda y: tf.gather(y[0], y[1]), name="pos_xyz_models", output_shape=(3, N,))(
        [input_models, positive_class_ids])  # [pos_ix, 3, N]
    # y_true_t_shape = tf.shape(y_true_t) # shape: [pos_ix, 3]
    # y_true_r_shape = tf.shape(y_true_r)  # shape: [pos_ix, 3, 3]
    # y_pred_t_shape = tf.shape(y_pred_t)  # shape: [pos_ix, 3]
    # y_pred_r_shape = tf.shape(y_pred_r)  # shape: [pos_ix, 3, 3]
    # pos_xyz_models_shape = tf.shape(pos_xyz_models)  # [pos_ix, N, 3]
    # print_op = tf.print([y_pred_r_shape,y_true_r_shape, y_pred_t_shape, y_pred_r_shape, pos_xyz_models_shape])

    # with tf.control_dependencies([print_op]):
    loss = KL.Lambda(lambda y: tf.cond(tf.greater(tf.size(y[2]), tf.constant(0)),
                                       lambda: chamfer_distance_loss(y[0], y[1], y[2], y[3], y[4]),
                                       lambda: tf.constant(0.0)), name="loss")(
        [y_pred_r, y_pred_t, y_true_r, y_true_t, pos_xyz_models])
    # loss = K.mean(loss)
    model = KE.Model(inputs=[input_poses, input_class_ids, input_trans, input_rot, input_models],
                     outputs=[target_trans, target_rot, pred_trans, pred_rot, positive_ix, positive_class_ids, indices,
                              y_true_t, y_true_r, y_pred_t, y_pred_r, pos_xyz_models, loss])
    return model


##### from Frustrum
# is used as:
#     center_dist = tf.norm(center_label - end_points['center'], axis=-1)
#     center_loss = huber_loss(center_dist, delta=2.0)
def huber_loss(error, delta):
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return tf.reduce_mean(losses)


class ChamferLossTest(tf.test.TestCase):

    def testRotationConvergence(self):
        with open(os.path.join(HOME_FOLDER, "Code/Python/Mask_RCNN/samples/YCB_Video/XYZ_Models.pkl"), "rb") as f:
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
                trainloss, _ = sess.run([chamfer, train])
                # if counter == 1000:
                #     sess.run(print_op)
                #     counter = 0
            self.assertAllClose(tf_rot1.eval()[1:], tf_rot2.eval()[1:], rtol=1e-4)

    def testTranslationConvergence(self):
        with open(os.path.join(HOME_FOLDER, "Code/Python/Mask_RCNN/samples/YCB_Video/XYZ_Models.pkl"), "rb") as f:
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
                trainloss, _ = sess.run([chamfer, train])
                # if counter == 10:
                #     sess.run(print_op)
                #     counter = 0
            self.assertAllClose(tf_trans1.eval()[1:], tf_trans2.eval()[1:], rtol=1e-4)

    def testPoseLossGraph(self):
        # TODO: Do it
        return True


if __name__ == '__main__':
    from samples.LINEMOD.LINEMOD import linemod_point_cloud
    from samples.YCB_Video.YCB_Video import YCBVDataset

    HOME_FOLDER = "/common/homes/staff/pohl/"
    YCB_path = os.path.join(HOME_FOLDER, "Hitachi/YCB_Video_Dataset/")
    xyz_file = "models/035_power_drill/points.xyz"
    print(f"Optimizing Loss for {xyz_file}")
    dataset = YCBVDataset()
    dataset.load_ycbv(YCB_path, "trainval")
    dataset.prepare()
    image = dataset.load_image(dataset.image_from_source_map["YCBV.0081/000982"])
    mask, classes = dataset.load_mask(dataset.image_from_source_map["YCBV.0081/000982"])
    poses, pclasses = dataset.load_pose(dataset.image_from_source_map["YCBV.0081/000982"])
    df = create_xyz_dataframe(dataset, YCB_path)
    with open("/common/homes/staff/pohl/Code/Python/Mask_RCNN/samples/YCB_Video/XYZ_Models.pkl", "wb") as f:
        pkl.dump(df, f)
    tf.test.main()
