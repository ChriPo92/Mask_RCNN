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
        # def min_nonzero(tensor):
        #     """
        #     computes the minimum value of a tensor not including values that are 0
        #     :param tensor:
        #     :return:
        #     """
        #     mask = tf.greater(tensor, 0., name="depth_mask")
        #     masked_tensor = tf.boolean_mask(tensor, mask)
        #     return tf.reduce_min(masked_tensor)
        # [batch, num_rois, (y1, x1, y2, x2)], [batch, num_rois, 24, 24, config.FEATURE_PYRAMIND_TOP_DOWN_SIZE]
        # [batch, num_rois, 24, 24, 1], [batch, 3, 3]
        rois, image_features, depth_image, intrinsic_matrices = inputs
        # [batch, num_rois, 1], [batch, num_rois, 1], [batch, num_rois, 1], [batch, num_rois, 1]
        y1, x1, y2, x2 = tf.split(rois, 4, axis=2)
        im_shape = tf.shape(image_features)
        # TODO: this works only as long as the rois have shape 24 x 24
        batch = self.config.BATCH_SIZE
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
        # min_z = tf.reshape(utils.batch_slice([z_im], min_nonzero,
        #                                      self.config.IMAGES_PER_GPU,
        #                                      names=["min_nonzero_z"]),
        #                    (batch, 1, 1, 1))
        # rescales z to be between 0...1 for each batch, excluding the depth points == 0
        # reshapes z_im to [batch, num_rois*w*h], then takes then min/max along dimension 1
        # to result in an tensor of shape [batch]
        # sub1 = tf.subtract(
        #         z_im,
        #         min_z
        #     )
        # sub2 = tf.subtract(
        #     tf.reshape(
        #         tf.reduce_max(
        #             tf.reshape(z_im, (batch, -1)),
        #             axis=1),
        #         (batch, 1, 1, 1)),
        #         min_z
        #     )

        # z_im = tf.div(sub1, sub2)
        # [batch, num_rois, h*w, 3]

        positions = tf.concat([x_im, y_im, z_im], axis=-1, name="concat_positions")
        positions = tf.reshape(positions, (batch, num_rois, -1, 3))
        # print_op = tf.print([tf.shape(y_im), tf.shape(x_im), tf.shape(z_im), tf.shape(positions)])
        # with tf.control_dependencies([print_op]):
        # [batch, num_rois, h*w, config.FEATURE_PYRAMID_TOP_DOWN_SIZE]
        features = tf.reshape(image_features, (batch, num_rois, -1, channels), name="reshape_features")
        return [positions, features]

    def compute_output_shape(self, input_shape):
        rois_shape = input_shape[0]
        image_shape = input_shape[1]
        feature_maps = self.feature_map_size[0] * self.feature_map_size[1]
        output_shape = [(rois_shape[:2]) +(feature_maps, 3),
                        image_shape[:2] + (feature_maps, image_shape[-1])]
        return output_shape

def build_PointNet_Keras_Graph(point_cloud_tensor, num_points, config, train_bn,
                               name, out_number, last_activation="linear", vector_size=1024):
    """

    :param point_cloud_tensor: [batch, num_rois, classes, num_points, 7, 1]
    :param num_points: num_points extracted from masks
    :param train_bn:
    :param name:
    :param out_number:
    :param last_activation:
    :param vector_size: size of the pointnet vector
    :return: [batch, num_rois, num_classes, out_number]
    """
    # transform to [batch, num_rois * num_classes, num_points, 7, 1]
    # point_cloud_tensor = KL.Lambda(lambda y: tf.transpose(y, []))(point_cloud_tensor)
    # point_cloud_tensor = KL.Lambda(lambda y: tf.reshape(y, (config.BATCH_SIZE, -1, num_points,
    #                                                         7, 1)))(point_cloud_tensor)
    # transform to [batch, num_rois * num_classes, num_points, 6, 16]
    class_partials = []
    # we don't want to use TimeDistributed here, since it uses the same weights for each
    # temporal slice. Therefore we switch to a for loop, which is not a perfect solution.
    # The rois are handled as seperate columns of an "image", where only the columns belonging
    # to one roi are used together (i.e. in the first step the kernel is of size (1, 7) and
    # the step size is (also 1, 7)
    # extract for each class [batch, num_rois, num_points, 7, 1]
    # partial_point_cloud = KL.Lambda(lambda y: y[:, :, j, :, :])(point_cloud_tensor)
    # transpose to [batch, num_classes, num_points, num_rois, 7, 1] and reshape to [batch, num_classes, num_points, num_rois * 7, 1]
    partial_point_cloud = KL.Lambda(lambda y: tf.reshape(tf.transpose(y, [0, 2, 3, 1, 4, 5]),
                                                         (config.BATCH_SIZE * config.NUM_CLASSES, num_points,
                                                          config.TRAIN_ROIS_PER_IMAGE * 7, 1)))(point_cloud_tensor)
    # transform to [batch * num_classes, num_points, num_rois, 64]
    x = KL.Conv2D(64, (1, 7), strides=(1, 7), padding="valid",
                           name=f"mrcnn_pointnet_{name}_conv1")(partial_point_cloud)
    x = KL.BatchNormalization(
                           name=f'mrcnn_pointnet_{name}_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # # transform to [batch, num_rois * num_classes, num_points, 4, 32]
    # x = KL.Conv2D(32, (1, 3), padding="valid",
    #               name=f"mrcnn_pointnet_{name}_conv2")(x)
    # x = KL.BatchNormalization(name=f'mrcnn_pointnet_{name}_bn2')(x, training=train_bn)
    # x = KL.Activation('relu')(x)
    # # transform to [batch, num_rois * num_classes, num_points, 1, 64]
    # x = KL.Conv2D(64, (1, 4), padding="valid",
    #               name=f"mrcnn_pointnet_{name}_conv3")(x)
    # x = KL.BatchNormalization(name=f'mrcnn_pointnet_{name}_bn3')(x, training=train_bn)
    #
    # x = KL.Activation('relu')(x)
    # transform to [batch, num_classes, num_points, num_rois, 128]
    x = KL.Conv2D(128, (1, 1), padding="valid",
                  name=f"mrcnn_pointnet_{name}_conv4")(x)
    x = KL.BatchNormalization(
                           name=f'mrcnn_pointnet_{name}_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # transform to [batch, num_classes, num_points, num_rois, vector_size]
    x = KL.Conv2D(vector_size, (1, 1), padding="valid",
                  name=f"mrcnn_pointnet_{name}_conv5")(x)
    x = KL.BatchNormalization(
                           name=f'mrcnn_pointnet_{name}_bn5')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # transform to [batch, num_classes, 1, num_rois, vector_size]
    x = KL.MaxPool2D((num_points, 1), padding="valid",
                     name=f"mrcnn_{name}_sym_max_pool")(x)
    # transform to [batch, num_classes, num_rois, vector_size]
    # and transpose to [batch, num_rois, num_classes, vector_size]
    x = KL.Lambda(lambda y: tf.transpose(tf.reshape(y, (config.BATCH_SIZE,
                                                        config.NUM_CLASSES,
                                                        config.TRAIN_ROIS_PER_IMAGE,
                                                        vector_size)), [0, 2, 1, 3]))(x)
    # transform to [batch * num_rois, num_classes, vector_size] so that all classes have their own weigths,
    # but all rois have the same weights
    x = KL.Lambda(lambda y: tf.reshape(y, (config.BATCH_SIZE * config.TRAIN_ROIS_PER_IMAGE,
                                           config.NUM_CLASSES, vector_size)))(x)
    # transform to [batch * num_rois, num_classes, 256]
    x = KL.Dense(256,
                 name=f"mrcnn_pointnet_{name}_fc1")(x)
    x = KL.BatchNormalization(
                           name=f'mrcnn_pointnet_{name}_bn6')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # transform to [batch, num_rois, num_classes, 128]
    x = KL.Dense(128,
                 name=f"mrcnn_pointnet_{name}_fc2")(x)
    x = KL.BatchNormalization(
        name=f'mrcnn_pointnet_{name}_bn7')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # [batch, num_rois, num_classes, out_number]
    x = KL.Dense(out_number, activation=last_activation,
                 name=f"mrcnn_pointnet_{name}_fc3")(x)
    x = KL.Lambda(lambda y: tf.reshape(y, (config.BATCH_SIZE, config.TRAIN_ROIS_PER_IMAGE,
                                           config.NUM_CLASSES, out_number)))(x)
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

class SamplePointsFromMaskedRegion(KL.Layer):

    def __init__(self, num_points, **kwargs):
        super(SamplePointsFromMaskedRegion, self).__init__(**kwargs)
        self.num_points = num_points

    def call(self, inputs, **kwargs):
        """

        :param inputs: [xyz, features, masks]
            xyz: [batch, num_rois, pool_size * pool_size, 3, 1]
            features: [batch, num_rois, pool_size * pool_size, 4, 1]
            masks: [batch, num_rois, pool_size, pool_size, num_classes]
        :param kwargs:
        :return: [batch, num_rois, num_classes, num_points, 7, 1]
        """
        xyz, features, masks = inputs
        shape = tf.shape(masks)
        batch = shape[0]
        num_rois = shape[1]
        mask_area = shape[2] * shape[3]
        num_classes = shape[-1]
        # transform to [batch, num_rois, pool_size * pool_size, num_classes]
        with tf.control_dependencies([tf.assert_equal(tf.shape(xyz)[2], mask_area)]):
            masks = tf.reshape(masks, (batch, num_rois, -1, num_classes))
            # prevent pose estimation to influnce mask computation
            masks = tf.stop_gradient(masks)
        # [batch, num_rois, pool_size², num_classes]
        bool_mask = tf.where(masks >= 0.5, tf.ones_like(masks), tf.zeros_like(masks))
        #### Shuffle the points of the mask #####
        # otherwise the upper left corner of masks is strongly favored
        idx = tf.meshgrid(*[tf.range(s) for s in [batch, num_rois,
                                                  mask_area, num_classes]], indexing="ij")
        # stack the index-tensors to [batch, num_rois, mask_area == pool_size², num_classes, 4]
        idx = tf.stack(idx, axis=-1)
        # transpose to [pool_size², batch, num_rois, num_classes, 4]
        idx = tf.transpose(idx, [2, 0, 1, 3, 4])
        # shuffle the first dimension of idx (pool_size) and
        # transpose back to [batch, num_rois, pool_size², num_classes, 4]
        shuffle_idx = tf.transpose(tf.random.shuffle(idx), [1, 2, 0, 3, 4])
        # shuffle the masks along the points axis
        bool_mask = tf.gather_nd(bool_mask, shuffle_idx)
        # [batch, num_rois, pool_size², x, num_classes] -> [batch, num_rois, pool_size², num_classes, 3]
        tiled_xyz = tf.transpose(tf.tile(xyz, [1, 1, 1, 1, num_classes]), [0, 1, 2, 4, 3])
        # shuffle same way as bool_masks
        tiled_xyz = tf.gather_nd(tiled_xyz, shuffle_idx)
        # create a mask of the same shape as bool_mask with elements indicating (==1)
        # if the depth is nonzero, otherwise 0
        zero_depth_mask = tf.where(tiled_xyz[:, :, :, :, 2] > 0.0, tf.ones_like(masks), tf.zeros_like(masks))
        # transpose to [batch, num_rois, num_classes, pool_size², 3]
        tiled_xyz = tf.transpose(tiled_xyz, [0, 1, 3, 2, 4])
        # [batch, num_rois, pool_size², x, num_classes] -> [batch, num_rois, pool_size², num_classes, 4]
        tiled_features = tf.transpose(tf.tile(features, [1, 1, 1, 1, num_classes]), [0, 1, 2, 4, 3])
        # shuffle same way as bool_masks and transpose to [batch, num_rois, num_classes, pool_size², 4]
        tiled_features = tf.transpose(tf.gather_nd(tiled_features, shuffle_idx), [0, 1, 3, 2, 4])
        # multiply bool_mask with zero_depth_mask, to get only elements that have nonzero depth
        # in the resulting mask
        bool_mask = tf.multiply(zero_depth_mask, bool_mask)
        # select the top "num_points" entries in the boolean mask tensor; since
        # those are only 0 and 1, only not masked entries from the tensor are selected
        # if there are less than "num_points" that are not masked, it is filled with
        # other points
        bool_values, bool_idx = tf.nn.top_k(tf.transpose(bool_mask, [0, 1, 3, 2]),
                                            self.num_points, sorted=False)
        # construct a index tensor that is compatible with tf.gather_nd, which needs
        # the indices as the last dimension of the index-tensor
        # create three tensors of shape [batch, num_rois, num_classes] with the indices
        # to only one dimension
        idx = tf.meshgrid(*[tf.range(s) for s in [batch, num_rois, num_classes]], indexing="ij")
        # stack the index-tensors to [batch, num_rois, num_classes, 1, 3]
        idx = tf.expand_dims(tf.stack(idx, axis=-1), axis=3)
        # tile it "num_points" times to [batch, num_rois, num_class, num_points, 3]
        idx = tf.tile(idx, [1, 1, 1, self.num_points, 1])
        # concat with the indices of the points found by top_k;
        # [batch, num_rois, num_classes, num_points, 3]
        idx = tf.concat([idx, tf.expand_dims(bool_idx, axis=-1)], axis=-1)
        # multiply the values from top_k with the points; this makes masked points equal to 0
        tiled_xyz = tf.multiply(tf.expand_dims(bool_values, axis=-1),
                                tf.gather_nd(tiled_xyz, idx))
        # [batch, num_rois, num_classes, num_points, 3, 1]
        tiled_xyz = tf.expand_dims(tiled_xyz, axis=-1)

        tiled_features = tf.multiply(tf.expand_dims(bool_values, axis=-1),
                                     tf.gather_nd(tiled_features, idx))
        # [batch, num_rois, num_classes, num_points, 4, 1]
        tiled_features = tf.expand_dims(tiled_features, axis=-1)
        return tf.concat([tiled_xyz, tiled_features], axis=-2)

    def compute_output_shape(self, input_shape):
        batch = input_shape[0][0]
        num_rois = input_shape[0][1]
        num_classes = input_shape[2][-1]
        return (batch, num_rois, num_classes, self.num_points, 7, 1)



def build_fpn_pointnet_pose_graph(rois, feature_maps, depth_image, image_meta, masks, intrinsic_matrices,
                                    config, pool_size=18, train_bn=True):
    """Builds the computation graph of the pose estimation head of Feature Pyramid Network.

        rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
              coordinates.
        feature_maps: List of feature maps from different layers of the pyramid,
                      [P2, P3, P4, P5]. Each has a different resolution.
        image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        masks: [batch, num_rois, 2 * MASK_POOL_SIZE, 2 * MASK_POOL_SIZE, NUM_CLASSES] output from the mask_branch
        intrinsic_matrices: [batch, 3, 3] Intrinsic Matrix of the camera batch was taken with
        train_bn: Boolean. Train or freeze Batch Norm layers

        Returns: Trans [[batch, num_rois, 3, 1, num_classes]
        """
    num_points = 200
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
    # merge to [batch, num_rois, num_classes, num_points, 7, 1]
    concat_point_cloud = SamplePointsFromMaskedRegion(num_points,
                                                      name="point_cloud_repr_concat")([pcl_list, feature_list, masks])
    # concat_point_cloud = KL.Lambda(lambda y: tf.concat([y[0], y[1]], axis=-2),
    #                              name="point_cloud_repr_concat")([pcl_list, feature_list])
    if config.POSE_ESTIMATION_METHOD is "pointnet2":
        assert False
        concat_point_cloud = build_PointNet2_Feature_Graph(concat_point_cloud,
                                                           train_bn, 0.5)
        trans = build_PointNet2_Regr_Graph(concat_point_cloud, pool_size, train_bn, "trans",
                                         3 * config.NUM_CLASSES)
        rot = build_PointNet2_Regr_Graph(concat_point_cloud, pool_size, train_bn, "rot",
                                         6 * config.NUM_CLASSES, last_activation="tanh")
    else:
        # [batch, num_rois, num_classes, 3]
        trans = build_PointNet_Keras_Graph(concat_point_cloud, num_points, config,
                                           train_bn, "trans", 3,
                                           vector_size=config.POINTNET_VECTOR_SIZE)
        # transform to [batch, num_rois, num_classes, 1, 3, 1]
        trans = KL.Reshape((config.TRAIN_ROIS_PER_IMAGE, config.NUM_CLASSES,
                            1, 3, 1), name="trans_preshape")(trans)
        pcl_list = KL.Lambda(lambda y: y[:, :, :, :, :3])(concat_point_cloud)
        feature_list = KL.Lambda(lambda y: y[:, :, :, :, 3:])(concat_point_cloud)
        pcl_list = KL.Subtract()([pcl_list, trans])
        concat_point_cloud = KL.Lambda(lambda y: tf.concat(y, axis=-2),
                                       name="centered_concat_point_clouds")([pcl_list, feature_list])
        rot = build_PointNet_Keras_Graph(concat_point_cloud, num_points, config,
                                         train_bn, "rot", 6,
                                         last_activation="tanh",
                                         vector_size=config.POINTNET_VECTOR_SIZE)

    # [batch, num_rois, 3, 1, num_classes]
    trans = KL.Lambda(lambda y: tf.transpose(tf.squeeze(y, axis=3), [0, 1, 3, 4, 2]), name="trans_reshape")(trans)
    # [batch, num_rois, 3, 2, num_classes]
    rot = KL.Reshape((config.TRAIN_ROIS_PER_IMAGE, config.NUM_CLASSES,
                     3, 2), name="rot_reshape")(rot)
    rot = KL.Lambda(lambda y: tf.transpose(y, [0, 1, 3, 4, 2]))(rot)
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