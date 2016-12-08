# model_compression
Implementation of model compression with knowledge distilling method.

##### cifar-10 
Download cifar-10 dataset
> wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

The teacher model is pre-trained by caffe. And extract the model weight by [3]

This the work, I use network in network as teacher model. lenet as student model.

"teacher.npy" is the pre-trained model weights of teacher model[4].
"student.npy" is the model weights train on lenet directly.


# References
[1] Ba, J. and Caruana, R. Do deep nets really need to be deep? In NIPS 2014.
[2] Bharat Bhusan Sau Vineeth N. Balasubramanian, Deep Model Compression: Distilling Knowledge from Noisy Teachers. arXiv 2016.
[3] https://github.com/ethereon/caffe-tensorflow
[4] Network in Network model - https://github.com/aymericdamien/TensorFlow-Examples/


