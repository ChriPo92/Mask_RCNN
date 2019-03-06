from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
interpolate_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_interpolate_so.so'))
def three_nn(xyz1, xyz2):
    '''
    Input:
        xyz1: (b,n,3) float32 array, unknown points
        xyz2: (b,m,3) float32 array, known points
    Output:
        dist: (b,n,3) float32 array, distances to known points
        idx: (b,n,3) int32 array, indices to known points
    '''
    return interpolate_module.three_nn(xyz1, xyz2)
ops.NoGradient('ThreeNN')
def three_interpolate(points, idx, weight):
    '''
    Input:
        points: (b,m,c) float32 array, known points
        idx: (b,n,3) int32 array, indices to known points
        weight: (b,n,3) float32 array, weights on known points
    Output:
        out: (b,n,c) float32 array, interpolated point values
    '''
    return interpolate_module.three_interpolate(points, idx, weight)
@tf.RegisterGradient('ThreeInterpolate')
def _three_interpolate_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    weight = op.inputs[2]
    return [interpolate_module.three_interpolate_grad(points, idx, weight, grad_out), None, None]


def kNN(xyz1, xyz2, k):
    """
    Calculates the indices and distances of the k nearest neighbor points in xyz2
    to every point in xyz1

    :param xyz1: [b, n, 3] first pointset
    :type xyz1: tf.Tensor
    :param xyz2: [b, m, 3] second pointset
    :type xyz2: tf.Tensor
    :param k: number of nearest neighbors to find
    :type k: int
    :return: (distances, indices):

        indices: [b, n, k] indices of the k nearest neighbors for each point in :param xyz1:
        distances: [b, n, k] euclidean distances of the k nearest neighbors for each point in :param xyz1:

    :rtype: (tf.Tensor, tf.Tensor)

    """
    n = tf.shape(xyz1)[1]
    m = tf.shape(xyz2)[1]
    # [m, b, 1, 3]
    xyz2_t = tf.expand_dims(tf.transpose(xyz2, [1, 0, 2]), axis=2)
    # [m, b, n, 3]
    txyz2 = tf.tile(xyz2_t, [1, 1, n, 1])
    squared_diff = tf.squared_difference(xyz1, txyz2)
    squared_dist = tf.reduce_sum(squared_diff, axis=-1)
    dist = tf.sqrt(squared_dist)
    # if there are less points in xyz2 then k, top_k does not work
    k_min = tf.minimum(k, m)
    # [b, n, k_min], [b, n, k_min]
    value, idx = tf.nn.top_k(tf.negative(tf.transpose(dist, [1, 2, 0])), k_min)
    value = tf.negative(value)
    value = tf.cond(m > k, lambda: value, lambda: tf.tile(value, [1, 1, tf.ceil(k / m)])[:, :, :k])
    idx = tf.cond(m > k, lambda: idx, lambda: tf.tile(idx, [1, 1, tf.ceil(k / m)])[:, :, :k])
    return value, idx

def interpolate_kNN(points, indices, weights):
    """

    :param points: [b,m,c] known points
    :type points: tf.Tensor
    :param indices: [b, n, k] indices to known points
    :type indices: tf.Tensor
    :param weights: [b, n, k] weights on known points
    :type weights: tf.Tensor
    :return:
    :rtype:
    """
    shape = tf.shape(points)
    m, c = shape[1], shape[2]
    shape = tf.shape(indices)
    b, n, k = shape[0], shape[1], shape[2]

    idx = tf.reshape(tf.range(b), (b, 1, 1))
    # [b, c, 2]
    # idx = tf.expand_dims(idx, axis=1)
    # [b, n, k]
    idx = tf.tile(idx, [1, n, k])
    idx = tf.stack([idx, indices], axis=-1)
    new_points = tf.gather_nd(points, idx)
    new_points = tf.multiply(new_points, tf.expand_dims(weights, axis=-1))
    new_points = tf.reduce_sum(new_points, axis=2)
    return new_points



if __name__=='__main__':
    import numpy as np
    import time
    np.random.seed(100)
    pts = np.random.random((32,128,64)).astype('float32')
    tmp1 = np.random.random((32,512,3)).astype('float32')
    tmp2 = np.random.random((32,128,3)).astype('float32')
    with tf.device('/cpu:0'):
        points = tf.constant(pts)
        xyz1 = tf.constant(tmp1)
        xyz2 = tf.constant(tmp2)
        dist1, idx1 = three_nn(xyz1, xyz2)
        weight = tf.ones_like(dist1)/3.0
        interpolated_points = three_interpolate(points, idx1, weight)
    with tf.device("/gpu:0"):
        dist2, idx2 = kNN(xyz1, xyz2, 3)
        int_points2 = interpolate_kNN(points, idx2, weight)
    with tf.Session('') as sess:
        now = time.time() 
        for _ in range(1000):
            ret1 = sess.run(interpolated_points)
        print(time.time() - now)
        print(ret1.shape, ret1.dtype)
        now = time.time()
        for _ in range(1000):
            ret2 = sess.run(int_points2)
        print(time.time() - now)
        print(ret2.shape, ret2.dtype)
        np.testing.assert_allclose(ret1, ret2, atol=0.001)
        #print ret
    
    
    
