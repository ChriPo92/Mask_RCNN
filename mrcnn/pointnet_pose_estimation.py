import keras.layers as KL
import keras.engine as KE
import tensorflow as tf

from utils import utils, keras_util, tf_util
from utils.pointnet_util import pointnet_sa_module_msg, pointnet_sa_module, pointnet_fp_module

class FeaturePointCloud(KE.Layer):

    def __init__(self, feature_map_size, config, **kwargs):
        """

        :param rois: batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
                     coordinates
        :param image_features: [batch, num_rois, h, w, channels]
        :param depth_image: [batch, num_rois, h, w, 1]
        :return: [batch, num_rois, h*w, (x, y, z)], [batch, num_rois, h*w, channels]
        """
        super(FeaturePointCloud, self).__init__(**kwargs)
        self.feature_map_size = feature_map_size
        self.config = config

    def call(self, inputs, **kwargs):
        def min_nonzero(tensor):
            """
            computes the minimum value of a tensor not including values that are 0
            :param tensor:
            :return:
            """
            mask = tf.greater(tensor, 0., name="depth_mask")
            masked_tensor = tf.boolean_mask(tensor, mask)
            return tf.reduce_min(masked_tensor)
        # [batch, num_rois, (y1, x1, y2, x2)], [batch, num_rois, 24, 24, config.FEATURE_PYRAMIND_TOP_DOWN_SIZE]
        # [batch, num_rois, 24, 24, 1], [batch, 3, 3]
        rois, image_features, depth_image, intrinsic_matrices = inputs
        # [batch, num_rois, 1], [batch, num_rois, 1], [batch, num_rois, 1], [batch, num_rois, 1]
        y1, x1, y2, x2 = tf.split(rois, 4, axis=2)
        im_shape = tf.shape(image_features)
        # TODO: this works only as long as the rois have shape 24 x 24
        batch = self.config.IMAGES_PER_GPU
        num_rois = im_shape[1]
        h = tf.cast(self.feature_map_size[0], "float32")
        w = tf.cast(self.feature_map_size[1], "float32")
        channels = im_shape[4]
        # print_op = tf.print([batch, num_rois, h, w, channels])
        # with tf.control_dependencies([print_op]):
        # [width]
        x = tf.range(w)
        # [height]
        y = tf.range(h)
        # [h, w], [h, w]
        X, Y = tf.meshgrid(x, y)
        # [batch*h, w]; needs to be [batch, 1, h*w]
        X = tf.tile(X, [batch, 1], name="tiled_X")
        X = tf.reshape(X, (batch, 1, -1), name="reshape_X")
        # [batch, 1, h * w]
        # [1, h*w]
        Y = tf.tile(Y, [batch, 1], name="tiled_X")
        Y = tf.reshape(Y, (batch, 1, -1), name="reshape_Y")
        # [batch, 1, h * w]
        # [batch, num_rois, 1]
        height_pixel_scale = (y2 - y1) / h
        # [batch, num_rois, 1]
        width_pixel_scale = (x2 -x1) / w
        # expand_dims([batch, num_rois, 1] + [batch, num_rois, h*w]) =! [batch, num_rois, h*w, 1]
        z_im = tf.reshape(depth_image, (batch, num_rois, -1, 1), name="z_im")
        y_im = tf.expand_dims(y1 + tf.linalg.matmul(height_pixel_scale, Y), axis=-1)
        # [batch, num_rois, h*w, 1] - [batch, 1, 1, 1] = [batch, num_rois, h*w, 1] -> subtract principal point
        y_im = (y_im * self.config.IMAGE_SHAPE[0] - tf.reshape(intrinsic_matrices[:, 1, 2], (batch, 1, 1, 1)))
        # [batch, num_rois, h*w, 1] * [batch, num_rois, h*w, 1] = [batch, num_rois, h*w, 1] -> get correct depth scaling
        y_im = y_im * z_im
        # [batch, num_rois, h*w, 1] / [batch, 1, 1, 1] = [batch, num_rois, h*w, 1] -> divide by focal length
        y_im = y_im / tf.reshape(intrinsic_matrices[:, 1, 1], (batch, 1, 1, 1))
        x_im = tf.expand_dims(x1 + tf.linalg.matmul(width_pixel_scale, X), axis=-1)
        # x_im = (x_im * self.config.IMAGE_SHAPE[1] - intrinsic_matrices[:, 0, 2]) * z_im / intrinsic_matrices[:, 0, 0]
        x_im = (x_im * self.config.IMAGE_SHAPE[1] - tf.reshape(intrinsic_matrices[:, 0, 2], (batch, 1, 1, 1)))
        # [batch, num_rois, h*w, 1] * [batch, num_rois, h*w, 1] = [batch, num_rois, h*w, 1] -> get correct depth scaling
        x_im = x_im * z_im
        # [batch, num_rois, h*w, 1] / [batch, 1, 1, 1] = [batch, num_rois, h*w, 1] -> divide by focal length
        x_im = x_im / tf.reshape(intrinsic_matrices[:, 0, 0], (batch, 1, 1, 1))
        # [batch, num_rois, h*w, 1]
        # filter out all elements which have z == 0 for the minimum value
        min_z = tf.reshape(utils.batch_slice([z_im], min_nonzero,
                                             self.config.IMAGES_PER_GPU,
                                             names=["min_nonzero_z"]),
                           (batch, 1, 1, 1))
        # rescales z to be between 0...1 for each batch, excluding the depth points == 0
        # reshapes z_im to [batch, num_rois*w*h], then takes then min/max along dimension 1
        # to result in an tensor of shape [batch]
        sub1 = tf.subtract(
                z_im,
                min_z
            )
        sub2 = tf.subtract(
            tf.reshape(
                tf.reduce_max(
                    tf.reshape(z_im, (batch, -1)),
                    axis=1),
                (batch, 1, 1, 1)),
                min_z
            )

        # z_im = tf.div(sub1, sub2)
        # [batch, num_rois, h*w, 3]

        positions = tf.concat([x_im, y_im, z_im], axis=-1, name="concat_positions")
        positions = tf.reshape(positions, (batch, num_rois, -1, 3))
        # [batch, num_rois, h*w, config.FEATURE_PYRAMID_TOP_DOWN_SIZE]
        # print_op = tf.print([tf.shape(y_im), tf.shape(x_im), tf.shape(z_im), tf.shape(positions)])
        # with tf.control_dependencies([print_op]):
        features = tf.reshape(image_features, (batch, num_rois, -1, channels), name="reshape_features")
        return [positions, features]

    def compute_output_shape(self, input_shape):
        rois_shape = input_shape[0]
        image_shape = input_shape[1]
        feature_maps = self.feature_map_size[0] * self.feature_map_size[1]
        output_shape = [(rois_shape[:2]) +(feature_maps, 3),
                        image_shape[:2] + (feature_maps, image_shape[-1])]
        return output_shape

def build_PointNet_Keras_Graph(point_cloud_tensor, pool_size, train_bn,
                               name, out_number, last_activation="linear", vector_size=1024):
    # transform to [batch, num_rois, h*w(576), 6, 16]
    x = KL.TimeDistributed(KL.Conv2D(16, (1, 2), padding="valid"),
                                    name=f"mrcnn_pointnet_{name}_conv1")(point_cloud_tensor)
    x = KL.TimeDistributed(KL.BatchNormalization(),
                                    name=f'mrcnn_pointnet_{name}_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # transform to [batch, num_rois, h*w(576), 4, 32]
    x = KL.TimeDistributed(KL.Conv2D(32, (1, 3), padding="valid"),
                                    name=f"mrcnn_pointnet_{name}_conv2")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(),
                                    name=f'mrcnn_pointnet_{name}_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # transform to [batch, num_rois, h*w(576), 1, 64]
    x = KL.TimeDistributed(KL.Conv2D(64, (1, 4), padding="valid"),
                                    name=f"mrcnn_pointnet_{name}_conv3")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(),
                                    name=f'mrcnn_pointnet_{name}_bn3')(x, training=train_bn)

    x = KL.Activation('relu')(x)
    # transform to [batch, num_rois, h*w(576), 1, 256]
    x = KL.TimeDistributed(KL.Conv2D(256, (1, 1), padding="valid"),
                                    name=f"mrcnn_pointnet_{name}_conv4")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(),
                                    name=f'mrcnn_pointnet_{name}_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # transform to [batch, num_rois, h*w(576), 1, vector_size]
    x = KL.TimeDistributed(KL.Conv2D(vector_size, (1, 1), padding="valid"),
                                    name=f"mrcnn_pointnet_{name}_conv5")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(),
                                    name=f'mrcnn_pointnet_{name}_bn5')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # transform to [batch, num_rois, 1, 1, vector_size]
    x = KL.TimeDistributed(KL.MaxPool2D((pool_size * pool_size, 1), padding="valid"),
                                    name=f"mrcnn_{name}_sym_max_pool")(x)
    # transform to [batch, num_rois, vector_size]
    x = KL.Lambda(lambda y: tf.squeeze(y, axis=[2, 3]))(x)
    # transform to [batch, num_rois, 256]
    x = KL.TimeDistributed(KL.Dense(256),
                               name=f"mrcnn_pointnet_{name}_fc1")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(),
                           name=f'mrcnn_pointnet_{name}_bn6')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # transform to [batch, num_rois, 128]
    x = KL.TimeDistributed(KL.Dense(128),
                           name=f"mrcnn_pointnet_{name}_fc2")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(),
                           name=f'mrcnn_pointnet_{name}_bn7')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # [batch, num_rois, out_number]
    x = KL.TimeDistributed(KL.Dense(out_number, activation=last_activation),
                               name=f"mrcnn_pointnet_{name}_fc3")(x)
    return x

class CalcRotMatrix(KL.Layer):
    def __init__(self, config, **kwargs):
        super(CalcRotMatrix, self).__init__(**kwargs)
        self.batch_shape = [config.BATCH_SIZE,
                            config.TRAIN_ROIS_PER_IMAGE,
                            config.NUM_CLASSES]
    def call(self, inputs, **kwargs):
        """
        Transformes a Matrix with 2 out of 3 column vectors of a rotation matrix available
        From:
        https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
        :param inputs: [batch, num_rois, 3, 2, num_classes]
        :param kwargs:
        :return:
        """
        # transpose to [batch, num_rois, num_classes, 2, 3]
        inputs = tf.transpose(inputs, [0, 1, 4, 3, 2], "transpose_inputs")
        # split the column vectors; [batch, num_rois, num_classes, 3]
        a, b = tf.split(inputs, 2, axis=3, name="split_inputs")
        a = tf.squeeze(a, axis=3, name="squeeze_a")
        b = tf.squeeze(b, axis=3, name="squeeze_b")
        # calculate the norm of the vectors
        a_norm = tf.sqrt(tf.reduce_sum(tf.square(a), axis=3, keepdims=True) + 1e-8, name="a_norm")
        b_norm = tf.sqrt(tf.reduce_sum(tf.square(b), axis=3, keepdims=True) + 1e-8, name="b_norm")
        # transfrom them into unit vectors
        a = a / a_norm
        b = b / b_norm
        # [batch, num_rois, num_classes, 1]
        # dot product of the last dimensions (3-vectors); cosine of angle
        c = tf.squeeze(
            tf.matmul(
                tf.expand_dims(a, -1), tf.expand_dims(b, -1),
                transpose_a=True),
            axis=-1)
        v = tf.linalg.cross(a, b)
        v_1 = v[:, :, :, 0]
        v_2 = v[:, :, :, 1]
        v_3 = v[:, :, :, 2]
        zero_element = tf.zeros_like(v_1)
        # skew-symmetric cross-product matrix of v; [batch, num_rois, num_classes, 9]
        skew_symmetric_v = tf.stack([zero_element, -v_3, v_2, v_3, zero_element,
                                     -v_1, -v_2, v_1, zero_element], axis=-1)
        # reshape to [batch, num_rois, num_classes, 3, 3]
        skew_symmetric_v = tf.reshape(skew_symmetric_v, self.batch_shape + [3, 3])
        identity_matrix = tf.eye(3, batch_shape=self.batch_shape)
        # the factor is 0 iff cos(a, b) = -1, i.e. when they point in opposite directions
        # [batch, num_rois, num_classes, 1]
        factor = tf.div_no_nan(tf.ones_like(c), 1 + c)
        # broadcast to [batch, num_rois, num_classes, 3, 3] for multiplication
        factor = tf.broadcast_to(tf.expand_dims(factor, -1), skew_symmetric_v.shape)
        rot = identity_matrix + skew_symmetric_v + tf.multiply(tf.matmul(skew_symmetric_v, skew_symmetric_v), factor)

        # [batch, num_rois, num_classes, 3, 3] --> [batch, num_rois, 3, 3, num_classes]
        # TODO: check if this is correct
        rot = tf.transpose(rot, [0, 1, 3, 4, 2])
        return rot

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 3, 3, input_shape[-1])

def build_fpn_pointnet_pose_graph(rois, feature_maps, depth_image, image_meta, intrinsic_matrices,
                                    config, pool_size=18, train_bn=True):
    """Builds the computation graph of the pose estimation head of Feature Pyramid Network.

        rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
              coordinates.
        feature_maps: List of feature maps from different layers of the pyramid,
                      [P2, P3, P4, P5]. Each has a different resolution.
        image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        intrinsic_matrices: [batch, 3, 3] Intrinsic Matrix of the camera batch was taken with
        train_bn: Boolean. Train or freeze Batch Norm layers

        Returns: Trans [[batch, num_rois, 3, 1, num_classes]
        """
    # TODO: as it seems, TRAIN_ROIS_PER_IMAGE (200) becomes DETECTION_MAX_INSTANCES (100) for inference
    # ROIAlign returning [batch, num_rois, 24, 24, channels] so that in the end a 4x4 matrix
    # is predicted for every class
    x, d = keras_util.PyramidROIAlign([pool_size, pool_size], depth_image=depth_image,
                           name="roi_align_pointnet_pose")([rois, image_meta] + feature_maps)
    # pcl_list = [batch, num_rois, h*w(576), (xq, y, z)], feature_list = [batch, num_rois, h*w, channels]
    # TODO: add masking from the previous branch; i.e. use predicted object masks here
    pcl_list, feature_list = FeaturePointCloud((pool_size, pool_size), config,
                                               name="FeaturePointCloud")([rois, x, d, intrinsic_matrices])
    # transfrom to [batch, num_rois, h*w, (x, y, z), 1]
    pcl_list = KL.Lambda(lambda y: tf.expand_dims(y, -1), name="expand_pcl_list")(pcl_list)
    # expand to [batch, num_rois, h*w, channels, 1]
    feature_list = KL.Lambda(lambda y: tf.expand_dims(y, -1), name="expand_feature_list")(feature_list)
    # transform to [batch, num_rois, h*w, 4, 1]
    feature_list = KL.TimeDistributed(KL.Conv2D(1, (1, int(config.TOP_DOWN_PYRAMID_SIZE / 4)),
                                                padding="valid",
                                                strides=(1, int(config.TOP_DOWN_PYRAMID_SIZE / 4))),
                                      name="mrcnn_pose_feature_conv")(feature_list)
    feature_list = KL.TimeDistributed(KL.BatchNormalization(),
                                      name='mrcnn_pose_feature_bn')(feature_list, training=train_bn)
    feature_list = KL.Activation('relu')(feature_list)
    # merge to [batch, num_rois, h*w(576), 7, 1]
    # print_op = tf.print([tf.shape(feature_list), tf.shape(pcl_list)])
    concat_point_cloud = KL.Lambda(lambda y: tf.concat([y[0], y[1]], axis=-2),
                                 name="point_cloud_repr_concat")([pcl_list, feature_list])
    if config.POSE_ESTIMATION_METHOD is "pointnet2":
        concat_point_cloud = build_PointNet2_Feature_Graph(concat_point_cloud,
                                                           train_bn, 0.5)
        trans = build_PointNet2_Regr_Graph(concat_point_cloud, pool_size, train_bn, "trans",
                                         3 * config.NUM_CLASSES)
        rot = build_PointNet2_Regr_Graph(concat_point_cloud, pool_size, train_bn, "rot",
                                         6 * config.NUM_CLASSES, last_activation="tanh")
    else:
        trans = build_PointNet_Keras_Graph(concat_point_cloud, pool_size, train_bn, "trans",
                                           3 * config.NUM_CLASSES,
                                           vector_size=config.POINTNET_VECTOR_SIZE)
        rot = build_PointNet_Keras_Graph(concat_point_cloud, pool_size, train_bn, "rot",
                                         6 * config.NUM_CLASSES, last_activation="tanh",
                                         vector_size=config.POINTNET_VECTOR_SIZE)
    # transform to [batch, num_rois, 3, 1, num_classes]
    trans = KL.Reshape((config.TRAIN_ROIS_PER_IMAGE,
                        3, 1, config.NUM_CLASSES), name="trans_reshape")(trans)
    # [batch, num_rois, 3, 2, num_classes]
    rot = KL.Reshape((config.TRAIN_ROIS_PER_IMAGE,
                     3, 2, config.NUM_CLASSES), name="rot_reshape")(rot)
    # [batch, num_rois, 3, 3, num_classes]; uses orthogonality of rotation matrices to calc
    # the third column vector
    rot = CalcRotMatrix(name="CalcRotMatrix", config=config)(rot)

    # print_op = tf.print([tf.shape(feature_list), tf.shape(pcl_list), tf.shape(point_cloud_repr),
    #                      tf.shape(x), tf.shape(shared), rot, trans])
    # with tf.control_dependencies([print_op]):
    #     rot = KL.Lambda(lambda y: tf.identity(y), name="pose_identity_op")(rot)
    return trans, rot

########################################################################################################################
#                                   POINTNET++                                                                         #
########################################################################################################################

class MultiScaleGroupingSetAbstractionLayer(KL.Layer):
    def __init__(self, npoint, radius_list, nsample_list,
                 mlp_list, is_training, bn_decay, bn=True,
                 use_xyz=True, use_nchw=False, **kwargs):
        super(MultiScaleGroupingSetAbstractionLayer, self).__init__(**kwargs)
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.mlp_list = mlp_list
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.use_xyz = use_xyz
        self.use_nchw = use_nchw

    def call(self, inputs, **kwargs):
        """
        Set Abstraction with Multi-Scale Grouping
        :param inputs: [xyz, points]
            xyz: [batch, num_points, 3]
            points: [batch, num_points, channels]
        :param kwargs:
        :return: [new_xyz, new_points]
            new_xyz: [batch, npoint, 3]
            new_points: [batch, npoint, \sum_k{mlp[k][-1]}]
        """
        xyz = tf.slice(inputs, [0, 0, 0], [-1, -1, 3])
        points = tf.slice(inputs, [0, 0, 3], [-1, -1, -1])
        l_xyz, l_points = pointnet_sa_module_msg(xyz, points, npoint=self.npoint,
                               radius_list=self.radius_list, nsample_list=self.nsample_list,
                               mlp_list=self.mlp_list, is_training=self.is_training,
                               bn_decay=self.bn_decay, bn=self.bn, use_xyz=self.use_xyz,
                               use_nchw=self.use_nchw, scope=self.name)
        return tf.concat([l_xyz, l_points], axis=-1)

    def compute_output_shape(self, input_shape):
        batch = input_shape[0]
        channels = 0
        for l in self.mlp_list:
            channels += l[-1]
        return (batch, self.npoint, 3 + channels)

class SetAbstractionLayer(KL.Layer):
    def __init__(self, npoint, radius, nsample, mlp, mlp2,
                 group_all, is_training, bn_decay, bn=True,
                 pooling='max', knn=False, use_xyz=True,
                 use_nchw=False, **kwargs):
        super(SetAbstractionLayer, self).__init__(**kwargs)
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp = mlp
        self.mlp2 = mlp2
        self.group_all = group_all
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.pooling = pooling
        self.knn = knn
        self.use_xyz = use_xyz
        self.use_nchw = use_nchw

    def call(self, inputs, **kwargs):
        xyz = tf.slice(inputs, [0, 0, 0], [-1, -1, 3])
        points = tf.slice(inputs, [0, 0, 3], [-1, -1, -1])
        l_xyz, l_points, _ = pointnet_sa_module(xyz, points, npoint=self.npoint,
                                             radius=self.radius, nsample=self.nsample,
                                             mlp=self.mlp, mlp2=self.mlp2,
                                             group_all=self.group_all,
                                             is_training=self.is_training,
                                             bn_decay=self.bn_decay, scope=self.name,
                                             bn=self.bn, pooling=self.pooling,
                                             knn=self.knn, use_xyz=self.use_xyz,
                                             use_nchw=self.use_nchw)
        return tf.concat([l_xyz, l_points], axis=-1)

    def compute_output_shape(self, input_shape):
        batch = input_shape[0]
        channels = self.mlp2[-1] if self.mlp2 is not None else self.mlp[-1]
        return (batch, self.npoint, 3 + channels)

class FeaturePropagationLayer(KL.Layer):
    def __init__(self, mlp, is_training, bn_decay, bn=True, **kwargs):
        super(FeaturePropagationLayer, self).__init__(**kwargs)
        self.mlp = mlp
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn

    def call(self, inputs, **kwargs):
        xyz1 = tf.slice(inputs[0], [0, 0, 0, 0], [-1, -1, -1, 3])
        points1 = tf.slice(inputs[0], [0, 0, 0, 3], [-1, -1, -1, -1])
        xyz2 = tf.slice(inputs[1], [0, 0, 0, 0], [-1, -1, -1, 3])
        points2 = tf.slice(inputs[1], [0, 0, 0, 3], [-1, -1, -1, -1])
        num_rois = xyz1.get_shape()[1]
        points = []
        for i in range(num_rois):
            new_points = pointnet_fp_module(xyz1[:, i, :, :], xyz2[:, i, :, :], points1[:, i, :, :], points2[:, i, :, :],
                                            mlp=self.mlp, is_training=self.is_training,
                                            bn_decay=self.bn_decay, scope=self.name+f"_{i}", bn=self.bn)
            points.append(new_points)
        return tf.concat([xyz1, tf.stack(points, axis=1)], axis=-1)

    def compute_output_shape(self, input_shape):
        xyz1_shape = input_shape[0]
        return (xyz1_shape[0], xyz1_shape[1], xyz1_shape[2], 3 + self.mlp[-1])


def build_PointNet2_Feature_Graph(concat_points, is_training, bn_decay):
    # Set abstraction layers
    # [batch, num_rois, pool_size², 7, 1]
    concat_points = KL.Lambda(lambda y: tf.squeeze(y, axis=-1))(concat_points)
    # [batch, num_rois, 128, 323] = [batch, num_rois, 128, 3] + [batch, num_rois, 128, 320]
    l1_concat = KL.TimeDistributed(MultiScaleGroupingSetAbstractionLayer(128, [0.2, 0.4, 0.8],
                                                              [32, 64, 128],
                                                              [[32, 32, 64], [64, 64, 128], [64, 96, 128]],
                                                              is_training, bn_decay,
                                                              name='msg_layer1'))(concat_points)
    # [batch, num_rois, 32, 643] = [batch, num_rois, 32, 3] + [batch, num_rois, 32, 640]
    l2_concat = KL.TimeDistributed(MultiScaleGroupingSetAbstractionLayer(32, [0.4, 0.8, 1.6], [64, 64, 128],
                                                              [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
                                                              is_training, bn_decay,
                                                              name='msg_layer2'))(l1_concat)
    # [batch, num_rois, 1, 1027] = [batch, num_rois, 1, 3] + [batch, num_rois, 1, 1024]
    l3_concat = KL.TimeDistributed(SetAbstractionLayer(npoint=None, radius=None, nsample=None,
                                               mlp=[128, 256, 1024], mlp2=None, group_all=True,
                                               is_training=is_training, bn_decay=bn_decay,
                                               name='layer3'))(l2_concat)

    # Feature Propagation layers
    # l3_points = tf.concat([l3_points, tf.expand_dims(one_hot_vec, 1)], axis=2)
    # [batch, num_rois, 32, 131] = [batch, num_rois, 32, 3] + [batch, num_rois, 32, 128]
    l2_concat = FeaturePropagationLayer([128, 128], is_training, bn_decay,
                                        name='fa_layer1')([l2_concat, l3_concat])
    # [batch, num_rois, 128, 131] = [batch, num_rois, 128, 3] + [batch, num_rois, 128, 128]
    l1_concat = FeaturePropagationLayer([128, 128], is_training, bn_decay,
                                        name='fa_layer2')([l1_concat, l2_concat])
    # [batch, num_rois, pool_size², 131]
    l0_concat = FeaturePropagationLayer([128, 128], is_training, bn_decay,
                                        name='fa_layer3')([concat_points, l1_concat])
    # [batch, num_rois, pool_size², 128, 1]
    l0_points = KL.Lambda(lambda y: tf.expand_dims(tf.slice(y, [0, 0, 0, 3],
                                                            [-1, -1, -1, -1]),
                                                   axis=-1))(l0_concat)
    # FC layers
    # net = KL.TimeDistributed(KL.Lambda(lambda y: tf_util.conv1d(y, 128, 1, padding='VALID', bn=True,
    #                                          is_training=is_training,
    #                                          bn_decay=bn_decay), name='conv1d-fc1'))(l0_points)
    # net = KL.TimeDistributed(KL.Lambda(lambda y: tf_util.dropout(y, keep_prob=0.7,
    #                       is_training=is_training), name="dp1"))(net)
    # logits = KL.TimeDistributed(KL.Lambda(lambda y: tf_util.conv1d(y, out_number, 1, padding='VALID',
    #                                             activation_fn=None),
    #                    name="conv1d-fc2"))(net)
    return l0_points

def build_PointNet2_Regr_Graph(point_cloud_tensor, pool_size, train_bn,
                               name, out_number, last_activation="linear"):
    # transform to [batch, num_rois, h*w(576), 1, 64]
    x = KL.TimeDistributed(KL.Conv2D(128, (1, 128), padding="valid"),
                                    name=f"mrcnn_pointnet_{name}_conv1")(point_cloud_tensor)
    x = KL.TimeDistributed(KL.BatchNormalization(),
                                    name=f'mrcnn_pointnet_{name}_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(128, (1, 1), padding="valid"),
                           name=f"mrcnn_pointnet_{name}_conv2")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(),
                           name=f'mrcnn_pointnet_{name}_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # transform to [batch, num_rois, h*w(576), 1, 256]
    x = KL.TimeDistributed(KL.Conv2D(256, (1, 1), padding="valid"),
                                    name=f"mrcnn_pointnet_{name}_conv4")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(),
                                    name=f'mrcnn_pointnet_{name}_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # transform to [batch, num_rois, 1, 1, vector_size]
    x = KL.TimeDistributed(KL.MaxPool2D((pool_size * pool_size, 1), padding="valid"),
                                    name=f"mrcnn_{name}_sym_max_pool")(x)
    # transform to [batch, num_rois, vector_size]
    x = KL.Lambda(lambda y: tf.squeeze(y, axis=[2, 3]))(x)
    # transform to [batch, num_rois, 256]
    x = KL.TimeDistributed(KL.Dense(256),
                               name=f"mrcnn_pointnet_{name}_fc1")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(),
                           name=f'mrcnn_pointnet_{name}_bn6')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # transform to [batch, num_rois, 128]
    x = KL.TimeDistributed(KL.Dense(128),
                           name=f"mrcnn_pointnet_{name}_fc2")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(),
                           name=f'mrcnn_pointnet_{name}_bn7')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # [batch, num_rois, out_number]
    x = KL.TimeDistributed(KL.Dense(out_number, activation=last_activation),
                               name=f"mrcnn_pointnet_{name}_fc3")(x)
    return x