TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
CUDA_PATH=/usr/local/cuda

nvcc -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu \
	-I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 \
	-x cu -Xcompiler -fPIC -arch=sm_50 -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 -shared -o tf_nndistance_so.so tf_nndistance.cpp \
	tf_nndistance_g.cu.o -I $TF_INC -I$TF_INC/external/nsync/public -fPIC \
	-lcudart -lcublas -L $CUDA_PATH/lib64 -L$TF_LIB -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=0


