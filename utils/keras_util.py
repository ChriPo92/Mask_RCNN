import tensorflow as tf
import numpy as np
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import depth_aware_operations.da_convolution as da_conv
import depth_aware_operations.da_avg_pooling as da_avg_pool

DCKL = da_conv.keras_layers
DPKL = da_avg_pool.keras_layers

from utils import utils

class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """

    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)

def convolution_layer(input, filters, kernel_size, depth_image=None, strides=(1, 1), padding='valid', data_format=None,
                      dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                      bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                      activity_regularizer=None, kernel_constraint=None, bias_constraint=None, sim_factor=None,
                      **kwargs):
    image, depth = None, None
    if depth_image is None:
        image = KL.Conv2D(filters, kernel_size, strides=strides, padding=padding,
                          data_format=data_format, dilation_rate=dilation_rate, activation=activation,
                          use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                          kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                          activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                          bias_constraint=bias_constraint, **kwargs)(input)
        return image, depth

    else:
        # print("da_conv")
        image, depth = DCKL.DAConv2D(filters, kernel_size, strides=strides, padding=padding,
                          data_format=data_format, dilation_rate=dilation_rate, activation=activation,
                          use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                          kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                          activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                          bias_constraint=bias_constraint, return_depth=True, similarity_factor=sim_factor,
                                     **kwargs)([input, depth_image])
        return image, depth

def pooling_layer(input_image, pool_size, strides, padding="same", depth_image=None):
    image, depth = None, None
    if depth_image is None:
        image = KL.MaxPooling2D(pool_size, strides=strides, padding=padding)(input_image)
        return image, depth
    else:
        image, depth = DPKL.DAAveragePooling2D(depth_image=depth_image, pool_size=pool_size,
                                               strides=strides, padding=padding, return_depth=True)(input_image)
        return image, depth

############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True, depth=None):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x, d = convolution_layer(input_tensor, nb_filter1, (1, 1), depth_image=depth,
                          name=conv_name_base + '2a', use_bias=use_bias, sim_factor=8.3 / stage)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x, d = convolution_layer(x, nb_filter2, (kernel_size, kernel_size), depth_image=d,
                          padding='same', name=conv_name_base + '2b', use_bias=use_bias, sim_factor=8.3 / stage)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x, d = convolution_layer(x, nb_filter3, (1, 1), depth_image=d, name=conv_name_base + '2c',
                             use_bias=use_bias, sim_factor=8.3 / stage)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    if depth is not None:
        d = KL.Add()([d, depth])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x, d


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True, depth=None):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
        depth: depth "image" of the current convolved tensor
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x, d = convolution_layer(input_tensor, nb_filter1, (1, 1), strides=strides, depth_image=None,
                  name=conv_name_base + '2a', use_bias=use_bias, sim_factor=8.3 / stage)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x, d = convolution_layer(x, nb_filter2, (kernel_size, kernel_size), depth_image=None, padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias, sim_factor=8.3 / stage)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x, d = convolution_layer(x, nb_filter3, (1, 1), depth_image=None, name=conv_name_base +
                                           '2c', use_bias=use_bias, sim_factor=8.3 / stage)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut, s_d = convolution_layer(input_tensor, nb_filter3, (1, 1), depth_image=depth, strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias, sim_factor=8.3 / stage)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    # # TODO: Change this back when using depth convolutions in all parts?
    # if depth is not None:
    #     d = KL.Add()([d, s_d])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x, s_d


def resnet_graph(input_image, architecture, stage5=False, train_bn=True, depth_image=None):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    if depth_image is not None:
        d = KL.ZeroPadding2D((3, 3))(depth_image)
    else:
        d = None
    x, d = convolution_layer(x, 64, (7, 7), strides=(2, 2), name='res1_conv', use_bias=True, depth_image=d,
                             sim_factor=8.3)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x, d = pooling_layer(x, (3, 3), strides=(2, 2), padding="same", depth_image=d)
    C1 = x
    # Stage 2
    x, d = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn, depth=d)
    x, _ = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn, depth=None)
    x, _ = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn, depth=None)
    C2 = x
    # Stage 3
    x, d = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn, depth=d)
    x, _ = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn, depth=None)
    x, _ = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn, depth=None)
    x, _ = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn, depth=None)
    C3 = x
    # Stage 4
    x, d = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn, depth=d)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x, _ = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn, depth=None)
        # x = KL.Lambda(lambda y: tf.Print(y, [tf.shape(y)], message="This is the shape of x: "))(x)
    C4 = x
    # Stage 5
    if stage5:
        x, d = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn, depth=d)
        x, d = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn, depth=None)
        x, d = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn, depth=None)
        C5 = x
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


############################################################
#  Proposal Layer
############################################################

def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer(KE.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.

    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Anchors
        anchors = inputs[2]

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                            self.config.IMAGES_PER_GPU,
                                            names=["pre_nms_anchors"])

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = utils.batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors"])

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes,
                                  lambda x: clip_boxes_graph(x, window),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors_clipped"])

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-max suppression
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals

        proposals = utils.batch_slice([boxes, scores], nms,
                                      self.config.IMAGES_PER_GPU)
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


############################################################
#  ROIAlign Layer
############################################################

def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.log(x) / tf.log(2.0)


class PyramidROIAlign(KE.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, depth_image=None, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)
        self.depth_image=depth_image

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, TOP_DOWN_PYRAMID_SIZE]
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2, name="roi_align_area_split")
        h = tf.math.subtract(y2, y1, name="roi_height")
        w = tf.math.subtract(x2, x1, name="roi_width")
        # Use shape of first image. Images in a batch must have the same size.
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        # TODO: FIX NANs appearing in this tensor, How is it possible that h*w == 0; apparently this happens in gradient?
        area = tf.multiply(h, w, name="roi_area") + 1e-8
        with tf.control_dependencies([tf.assert_positive(area)]):
            roi_level = log2_graph(tf.sqrt(area, name="sqrt_area") / (224.0 / tf.sqrt(image_area,
                                                                    name="sqrt_image_area")))
        # roi_level = tf.where(area > 0, roi_level, tf.zeros_like(roi_level))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)), name="roi_level")
        roi_level = tf.squeeze(roi_level, 2, name="squeezed_roi_level")

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        if self.depth_image is not None:
            pooled_depth = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix, name=f"level_boxes_{i}")

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear", name=f"pooled_level_{i}"))
            # repeat the same operation for the depth input image, so that 3D
            # informations can be retained
            # method = "nearest" is taken here because there are a lot of faulty
            # pixels in the depth image that are 0, and with bilinear rescaling
            # these end upinfluence the pixel values too much

            if self.depth_image is not None:
                pooled_depth.append(tf.image.crop_and_resize(
                self.depth_image, level_boxes, box_indices, self.pool_shape,
                method="nearest", name=f"pooled_depth_level_{i}"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        if self.depth_image is not None:
            pooled_depth = tf.concat(pooled_depth, axis=0, name="pooled_depth")
        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0, name="box_to_level")
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1, name="box_range")
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1, name="box_to_level")

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0], name="top_k_ix").indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix, name="gathered_ix")
        pooled = tf.gather(pooled, ix, name="pooled_image")

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape, name="pooled")

        if self.depth_image is not None:
            pooled_depth = tf.gather(pooled_depth, ix)
            shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled_depth)[1:]], axis=0)
            pooled_depth = tf.reshape(pooled_depth, shape, name="pooled_depth")
            return [pooled, pooled_depth]
        return pooled

    def compute_output_shape(self, input_shape):
        if self.depth_image is not None:
            return [(input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)),
                    (input_shape[0][:2] + self.pool_shape + (1,))]
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)

############################################################
#  Detection Target Layer
############################################################

def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1, name="overlaps_graph_max1")
    x1 = tf.maximum(b1_x1, b2_x1, name="overlaps_graph_max2")
    y2 = tf.minimum(b1_y2, b2_y2, name="overlaps_graph_max3")
    x2 = tf.minimum(b1_x2, b2_x2, name="overlaps_graph_max4")
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, gt_poses, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
    proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
    gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.
    gt_poses: [4, 4, MAX_GT_INSTANCES] or None if config.ESTIMATE_6D_POSE is False

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
    masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
           boundaries and resized to neural network output size.
    poses: ONLY RETURNED IF config.ESTIMATE_6D_POSE is True
           [TRAIN_ROIS_PER_IMAGE, 4, 4] GT Poses corresponding to
           the masks
    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    if not config.ESTIMATE_6D_POSE:
        assert gt_poses is None
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                         name="trim_gt_masks")

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    crowd_masks = tf.gather(gt_masks, crowd_ix, axis=2)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)
    if config.ESTIMATE_6D_POSE:
        gt_poses = tf.gather(gt_poses, non_crowd_ix, axis=2)

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Compute overlaps with crowd boxes [proposals, crowd_boxes]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Assign positive ROIs to GT masks
    # Permute masks to [N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)
    if config.ESTIMATE_6D_POSE:
        transposed_poses = tf.expand_dims(tf.transpose(gt_poses, [2, 0, 1]), -1)
        roi_poses = tf.gather(transposed_poses, roi_gt_box_assignment)
        poses = tf.squeeze(roi_poses, axis=3)

    # Compute mask targets
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # Transform ROI coordinates from normalized image space
        # to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                     box_ids,
                                     config.MASK_SHAPE)
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])
    if config.ESTIMATE_6D_POSE:
        poses = tf.pad(poses, [[0, N + P], (0, 0), (0, 0)])
        return rois, roi_gt_class_ids, deltas, masks, poses
    else:
        return rois, roi_gt_class_ids, deltas, masks


class DetectionTargetLayer(KE.Layer):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]
        if self.config.ESTIMATE_6D_POSE:
            gt_poses = inputs[4]
        else:
            gt_poses = [None for i in range(self.config.BATCH_SIZE)]

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois", "target_class_ids", "target_bbox", "target_mask", "target_pose"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks, gt_poses],
            lambda v, w, x, y, z: detection_targets_graph(
                v, w, x, y, z, self.config),
            self.config.IMAGES_PER_GPU, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        if not self.config.ESTIMATE_6D_POSE:
            return [
                (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
                (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
                (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
                (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
                 self.config.MASK_SHAPE[1])  # masks
            ]
        else:
            return [
                (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
                (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
                (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
                (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
                 self.config.MASK_SHAPE[1]),  # masks
                (None, self.config.TRAIN_ROIS_PER_IMAGE, 4, 4) # poses
            ]

    def compute_mask(self, inputs, mask=None):
        if not self.config.ESTIMATE_6D_POSE:
            return [None, None, None, None]
        else:
            return [None, None, None, None, None]


############################################################
#  Detection Layer
############################################################

def refine_detections_graph(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
        coordinates are normalized.
    """
    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(
        rois, deltas_specific * config.BBOX_STD_DEV)
    # Clip boxes to image window
    refined_rois = clip_boxes_graph(refined_rois, window)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois, keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
            tf.gather(pre_nms_rois, ixs),
            tf.gather(pre_nms_scores, ixs),
            max_output_size=config.DETECTION_MAX_INSTANCES,
            iou_threshold=config.DETECTION_NMS_THRESHOLD)
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
    ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections


class DetectionLayer(KE.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = norm_boxes_graph(m['window'], image_shape[:2])

        # Run detection refinement graph on each item in the batch
        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
            self.config.IMAGES_PER_GPU)

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        return tf.reshape(
            detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)


############################################################
#  Region Proposal Network (RPN)
############################################################

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """
    # TODO: check if stride of 2 causes alignment issues if the feature map
    # is not even.
    # Shared convolutional base of the RPN
    shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """
    input_feature_map = KL.Input(shape=[None, None, depth],
                                 name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name="rpn_model")


############################################################
#  Feature Pyramid Network Heads
############################################################

def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, config, train_bn=True,
                         fc_layers_size=1024):
    """Builds the computation graph of the feature pyramid network classifier
    and regressor heads.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers

    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
    """
    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, TOP_DOWN_PYRAMID_SIZE]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_classifier")([rois, image_meta] + feature_maps)
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)),
                           name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       name="pool_squeeze")(x)

    # Classifier head
    # shape = K.int_shape(shared)
    dim1 = shared._keras_shape[1]
    # shape = KL.Lambda(lambda y: tf.shape(y))(shared)
    # shared = KL.Reshape((config.BATCH_SIZE * shape[1], fc_layers_size))(shared)
    shared = KL.Lambda(lambda y: tf.reshape(y, (config.BATCH_SIZE * dim1,
                                                fc_layers_size)))(shared)
    mrcnn_class_logits = KL.Dense(num_classes,
                                            name='mrcnn_class_logits')(shared)
    mrcnn_class_logits = KL.Lambda(lambda y: tf.reshape(y, (config.BATCH_SIZE,
                                                            dim1,
                                                            num_classes)))(mrcnn_class_logits)
    # mrcnn_class_logits = KL.Reshape((config.BATCH_SIZE,
    #                                  shape[1], num_classes))(mrcnn_class_logits)
    mrcnn_probs = KL.Activation("softmax",
                                     name="mrcnn_class")(mrcnn_class_logits)
    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = KL.Dense(num_classes * 4, activation='linear',
                           name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    mrcnn_bbox = KL.Lambda(lambda y: tf.reshape(y, (config.BATCH_SIZE,
                                                    dim1,
                                                    num_classes, 4)), name="mrcnn_bbox")(x)
    # mrcnn_bbox = KL.Reshape((config.BATCH_SIZE,
    #                                                 shape[1],
    #                                                 num_classes, 4))(x)
    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers

    Returns: Masks [batch, num_rois, 2 * MASK_POOL_SIZE, 2 * MASK_POOL_SIZE, NUM_CLASSES]
    """
    # ROI Pooling
    # Shape: [batch, num_rois, 2 * MASK_POOL_SIZE, 2 * MASK_POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_mask")([rois, image_meta] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_mask_deconv")(x)
    # print_op = tf.print([tf.shape(x)])
    # with tf.control_dependencies([print_op]):
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                           name="mrcnn_mask")(x)
    return x

def build_fpn_pose_graph(rois, feature_maps, depth_image, image_meta,
                         num_classes, train_bn=True):
    """Builds the computation graph of the pose estimation head of Feature Pyramid Network.

        rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
              coordinates.
        feature_maps: List of feature maps from different layers of the pyramid,
                      [P2, P3, P4, P5]. Each has a different resolution.
        image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        num_classes: number of classes, which determines the depth of the results
        train_bn: Boolean. Train or freeze Batch Norm layers

        Returns: Trans [[batch, num_rois, 3, 1, NUM_CLASSES],
                 Rot    [batch, num_rois, 3, 3, NUM_CLASSES]]
        """
    def time_distributed_da_conv_model(channels, kernel_size, padding, strides=(1, 1)):
        """
        keras.layers.TimeDistributed can only handle layers that have one input and one output
        tensor. To be able to use the DA Convolutions, the depth has to be concatenated to the
        image and then split before the op and re-concatenated afterwards. This is handled by
        a keras model.

        :param channels: channels of the output feature map
        :param kernel_size: size of the convolutional kernel
        :param padding: padding used in the convolutions
        :param strides: strides of the convolution kernel
        :return: keras Model
        """
        merged_input = KL.Input([24, 24, 257], name="merged_input")
        o_x, o_d = DCKL.DAConv2D(
            channels, kernel_size, padding=padding, strides=strides, return_depth=True
        )(
            KL.Lambda(
                lambda x: [x[:, :, :, :-1], tf.expand_dims(x[:, :, :, -1], axis=-1)]
            )(merged_input)
        )
        o_x = BatchNorm()(o_x, training=train_bn)
        o_x = KL.Activation("relu")(o_x)
        merged_output = KL.merge.concatenate([o_x, o_d], axis=-1)
        merged_model = KE.Model(inputs=merged_input, outputs=merged_output)
        return merged_model

    # ROIAlign returning [batch, num_rois, 24, 24, channels] so that in the end a 4x4 matrix
    # is predicted for every class
    x, d = PyramidROIAlign([24, 24], depth_image=depth_image,
                        name="roi_align_feature_pose")([rois, image_meta] + feature_maps)
    rois_trans = KL.Lambda(lambda y: tf.expand_dims(tf.expand_dims(y, axis=-1), axis=-1))(rois)
    rois_trans = KL.TimeDistributed(KL.Deconv2D(16, (1, 2), padding="valid"),
                                    name="mrcnn_pose_rois_trans_deconv")(rois_trans)
    rois_trans = KL.TimeDistributed(KL.Conv2D(num_classes, (2, 2), padding="valid", activation="tanh"),
                                    name="mrcnn_pose_rois_trans_conv")(rois_trans)
    # merge image and depth, so that x = x_d[:, :, :, :, :-1] & d = x_d[:, :, :, :, -1]
    # x_d = KL.merge.concatenate([x, d])
    # x_d = KL.TimeDistributed(time_distributed_da_conv_model(256, (3, 3), "same"),
    #                          name="mrcnn_pose_conv1")(x_d)
    # # x_d = KL.TimeDistributed(time_distributed_da_conv_model(256, (3, 3), "same"),
    # #                          name="mrcnn_pose_conv2")(x_d)
    # # x_d = KL.TimeDistributed(time_distributed_da_conv_model(256, (3, 3), "same"),
    # #                          name="mrcnn_pose_conv3")(x_d)
    # # halfes the width and height dimensions to [12, 12] --> [Batch, num_rois, 12, 12, 256]
    # x_d = KL.TimeDistributed(time_distributed_da_conv_model(256, (3, 3), "same", (2, 2)),
    #                          name="mrcnn_pose_conv4")(x_d)
    # # changes the width and height dimension to [6, 6] --> [Batch, num_rois, 6, 6, num_classes]
    # x_d = KL.TimeDistributed(time_distributed_da_conv_model(num_classes, (3, 3), "same", (2, 2)),
    #                          name="mrcnn_pose_conv5")(x_d)
    x_int = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                                    name="mrcnn_pose_conv1")(x)
    d_int = KL.TimeDistributed(KL.Conv2D(2, (3, 3), padding="same"),
                                    name="mrcnn_pose_dconv1")(d)
    x_int = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_pose_conv2")(x_int)
    d_int = KL.TimeDistributed(KL.Conv2D(4, (3, 3), padding="same"),
                           name="mrcnn_pose_dconv2")(d_int)
    x_int = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_pose_conv3")(x_int)
    d_int = KL.TimeDistributed(KL.Conv2D(8, (3, 3), padding="same"),
                           name="mrcnn_pose_dconv3")(d_int)
    d_add = KL.TimeDistributed(KL.Conv2D(8, (3, 3), padding="same"),
                               name="mrcnn_pose_dconv_add")(d)
    d = KL.Add(name="mrcnn_pose_res_depth")([d_int, d_add])
    x = KL.Add(name="mrcnn_pose_res_img")([x_int, x])
    # # halfes the width and height dimensions to [12, 12] --> [Batch, num_rois, 12, 12, 256]
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same", strides=(2, 2)),
                           name="mrcnn_pose_conv4")(x)
    d = KL.TimeDistributed(KL.Conv2D(16, (3, 3), padding="same", strides=(2, 2)),
                           name="mrcnn_pose_dconv4")(d)
    # changes the width and height dimension to [6, 6] --> [Batch, num_rois, 6, 6, num_classes]
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (3, 3), padding="same", strides=(2, 2)),
                           name="mrcnn_pose_conv5")(x)
    d = KL.TimeDistributed(KL.Conv2D(num_classes, (3, 3), padding="same", strides=(2, 2)),
                           name="mrcnn_pose_dconv5")(d)
    # shared = KL.Lambda(lambda y: y[:, :, :, :, :-1])(x_d) # discard the depth map
    shared = KL.Add(name="mrcnn_pose_img_depth_add")([x, d])
    # Translation regression
    # changes [h w] to [3 1]
    trans = KL.TimeDistributed(KL.Conv2D(num_classes, (2, 6), strides=2, activation="tanh"),
                           name="mrcnn_pose_trans_conva")(shared)
    trans = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="tanh"),
                           name="mrcnn_pose_trans_convb")(trans)
    trans = KL.Add()([trans, rois_trans])
    trans = KL.Lambda(lambda x: K.squeeze(x, 3),
                       name="mrcnn_pose_trans_squeeze")(trans)

    # Rotation regression
    # changes [h w] to [3 3]
    rot = KL.TimeDistributed(KL.Conv2D(num_classes, (4, 4), strides=1, activation="tanh"),
                           name="mrcnn_pose_rot_conva")(shared)
    rot = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="tanh"),
                               name="mrcnn_pose_rot_convb")(rot)

    return trans, rot

############################################################
#  Miscellenous Graph Functions
############################################################

def trim_zeros_graph(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


def denorm_boxes_graph(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [..., (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)

def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed tensors.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }
