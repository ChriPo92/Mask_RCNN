""" Compute Chamfer's Distance.

Original author: Haoqiang Fan.
Modified by Charles R. Qi
"""

import tensorflow as tf
from tensorflow.python.framework import ops
from keras.engine.base_layer import Layer
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
nn_distance_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_nndistance_so.so'))

def nn_distance(xyz1,xyz2):
    '''
Computes the distance of nearest neighbors for a pair of point clouds
input: xyz1: (batch_size,#points_1,3)  the first point cloud
input: xyz2: (batch_size,#points_2,3)  the second point cloud
output: dist1: (batch_size,#point_1)   distance from first to second
output: idx1:  (batch_size,#point_1)   nearest neighbor from first to second
output: dist2: (batch_size,#point_2)   distance from second to first
output: idx2:  (batch_size,#point_2)   nearest neighbor from second to first
    '''
    return nn_distance_module.nn_distance(xyz1,xyz2)
#@tf.RegisterShape('NnDistance')
#def _nn_distance_shape(op):
    #shape1=op.inputs[0].get_shape().with_rank(3)
    #shape2=op.inputs[1].get_shape().with_rank(3)
    #return [tf.TensorShape([shape1.dims[0],shape1.dims[1]]),tf.TensorShape([shape1.dims[0],shape1.dims[1]]),
        #tf.TensorShape([shape2.dims[0],shape2.dims[1]]),tf.TensorShape([shape2.dims[0],shape2.dims[1]])]
@ops.RegisterGradient('NnDistance')
def _nn_distance_grad(op,grad_dist1,grad_idx1,grad_dist2,grad_idx2):
    xyz1=op.inputs[0]
    xyz2=op.inputs[1]
    idx1=op.outputs[1]
    idx2=op.outputs[3]
    return nn_distance_module.nn_distance_grad(xyz1,xyz2,grad_dist1,idx1,grad_dist2,idx2)


class NNDistance(Layer):
    def __init__(self, **kwargs):
        super(NNDistance, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        Computes the distance of nearest neighbors for a pair of point clouds
        input: xyz1: (batch_size,#points_1,3)  the first point cloud
        input: xyz2: (batch_size,#points_2,3)  the second point cloud

        """
        if type(inputs) is not list or len(inputs) != 2:
            raise Exception('NNDistance must be called on a list of 2 tensors'
                            '. Got: ' + str(inputs))
        xyz1 = inputs[0]
        xyz2 = inputs[1]
        outputs = nn_distance(xyz1, xyz2)
        names = ["distance1", "indices1", "distance2", "indices2"]
        result = [tf.identity(o, name=n) for o, n in zip(list(outputs), names)]
        return result

    def get_output_shape_for(self, input_shape):
        """
        output: dist1: (batch_size,#points_1)   distance from first to second
        output: idx1:  (batch_size,#points_1)   nearest neighbor from first to second
        output: dist2: (batch_size,#points_2)   distance from second to first
        output: idx2:  (batch_size,#points_2)   nearest neighbor from second to first
        """
        assert type(input_shape) is list
        inp1, inp2 = input_shape
        num_points1 = inp1[1]
        num_points2 = inp2[1]
        batches = inp1[0]
        assert batches == inp2[0]
        shape1 = tuple([batches, num_points1])
        shape2 = tuple([batches, num_points2])
        return [shape1, shape1, shape2, shape2]

    def compute_output_shape(self, input_shape):
        assert type(input_shape) is list
        inp1, inp2 = input_shape
        num_points1 = inp1[1]
        num_points2 = inp2[1]
        batches = inp1[0]
        assert batches == inp2[0]
        shape1 = tuple([batches, num_points1])
        shape2 = tuple([batches, num_points2])
        return [shape1, shape1, shape2, shape2]


if __name__=='__main__':
    import numpy as np
    import random
    import time
    from tensorflow.python.ops import gradient_checker
    from tensorflow.python.ops.gradient_checker import compute_gradient
    from tensorflow.python.framework import dtypes, constant_op
    random.seed(100)
    np.random.seed(100)
    with tf.Session('') as sess:
        in_shape1 = (4, 2048, 3)
        in_shape2 = (4, 1024, 3)
        out_shape1 = (4, 2048)
        out_shape2 = (4, 1024)
        xyz1=np.random.random_sample(in_shape1).astype('float32')
        xyz2=np.random.random_sample(in_shape2).astype('float32')
        #with tf.device('/gpu:0'):
        if True:
            inp1=tf.Variable(xyz1)
            inp2=tf.constant(xyz2)
            reta,retb,retc,retd=nn_distance(inp1,inp2)
            loss=tf.reduce_sum(reta)+tf.reduce_sum(retc)
            train=tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)
        sess.run(tf.global_variables_initializer())
        t0=time.time()
        t1=t0
        best=1e100
        # for i in range(100):
        #     trainloss,_=sess.run([loss,train])
        #     newt=time.time()
        #     best=min(best,newt-t1)
        #     print(i,trainloss,(newt-t0)/(i+1),best)
        #     t1=newt


        # check gradient
        with tf.device('/cpu:0'):
            in1 = constant_op.constant(xyz1)
            in2 = constant_op.constant(xyz2)
            out1, out2, out3, out4 = nn_distance(in1, in2)
            err = gradient_checker.compute_gradient_error([in1, in2], [in_shape1, in_shape2],
                                                          out1, out_shape1)
            print(err) # 0.16938598453998566 (CPU/GPU)
            # err1 = gradient_checker.compute_gradient_error([in1, in2], [in_shape1, in_shape2],
            #                                               out2, out_shape1)
            # print(err1)
            err2 = gradient_checker.compute_gradient_error([in1, in2], [in_shape1, in_shape2],
                                                          out3, out_shape2)
            print(err2) # 0.12206334620714188 (CPU/GPU)
        # err3 = gradient_checker.compute_gradient_error([in1, in2], [in_shape1, in_shape2],
        #                                               out4, out_shape2)
        # print(err3)
        # err = gradient_checker.compute_gradient_error([in1, in2], [in_shape1, in_shape2],
        #                                               [out1, out2, out3, out4],
        #                                               [list(out_shape1), list(out_shape1),
        #                                                list(out_shape2), list(out_shape2)])
        #print sess.run([inp1,retb,inp2,retd])
        #grads=compute_gradient([inp1,inp2],[(16,32,3),(16,32,3)],loss,(1,),[xyz1,xyz2])
        #for i,j in grads:
            #print i.shape,j.shape,np.mean(np.abs(i-j)),np.mean(np.abs(i)),np.mean(np.abs(j))
        #for i in xrange(10):
            #t0=time.time()
            #a,b,c,d=sess.run([reta,retb,retc,retd],feed_dict={inp1:xyz1,inp2:xyz2})
            #print 'time',time.time()-t0
        #print a.shape,b.shape,c.shape,d.shape
        #print a.dtype,b.dtype,c.dtype,d.dtype
        #samples=np.array(random.sample(range(xyz2.shape[1]),100),dtype='int32')
        #dist1=((xyz1[:,samples,None,:]-xyz2[:,None,:,:])**2).sum(axis=-1).min(axis=-1)
        #idx1=((xyz1[:,samples,None,:]-xyz2[:,None,:,:])**2).sum(axis=-1).argmin(axis=-1)
        #print np.abs(dist1-a[:,samples]).max()
        #print np.abs(idx1-b[:,samples]).max()
        #dist2=((xyz2[:,samples,None,:]-xyz1[:,None,:,:])**2).sum(axis=-1).min(axis=-1)
        #idx2=((xyz2[:,samples,None,:]-xyz1[:,None,:,:])**2).sum(axis=-1).argmin(axis=-1)
        #print np.abs(dist2-c[:,samples]).max()
        #print np.abs(idx2-d[:,samples]).max()


