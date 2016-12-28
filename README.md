# model_compression
Implementation of model compression with three knowledge distilling or teacher student methods [1][2][3].<br>
The basic architecture is teacher-student model.

# cifar-10 
I used cifar-10 dataset to do this work.

Download cifar-10 dataset
> wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

# Implementation
In this the work, I use network in network[5] as teacher model, lenet[6] as student model.<br>
The teacher model is pre-trained by caffe. And extract the model weight by [4].<br>
Both network-in-network and lenet have little different from original model.<br>
In docs, there are two images for the network architecture.

"teacher.npy" is the pre-trained model weights of teacher model.

"student.npy" is the model weights train on lenet, using ground turth label directly.


#Usage
In teacher-student.py, there is three methods to train student network.

###Basic Usage
**train by [1]**
> python  teacher-student.py --task train --model savemodel

**train by [2]**
> python  teacher-student.py --task train --model savemodel --noisy [--noisy_ratio --noisy_sigma]

**train by [3]**
> python  teacher-student.py --task train --model savemodel --KD [--lamda --tau]

<br>
**testing**
>python  teacher-student.py --task test --model trained_model

<br>
**validation**
Also, you can validate your pre-trained teacher model by <br>
> python  teacher-student.py --task val

This can make sure that your caffe-teacher-model transfer to tensorflow successfully.
<br>
***python teacher-student.py -h *** for more information




# References
[1] Ba, J. and Caruana, R. Do deep nets really need to be deep? In NIPS 2014. 

[2] Bharat Bhusan Sau Vineeth N. Balasubramanian, Deep Model Compression: Distilling Knowledge from Noisy Teachers. arXiv 2016.

[3] Hinton, G. E., Vinyals, O., and Dean, J. Distilling the knowledge in a neural network. arXiv 2015.

[4] https://github.com/ethereon/caffe-tensorflow

[5] Network in Network model - https://github.com/aymericdamien/TensorFlow-Examples/

[6] Y. LeCun, L. Bottou, Y. Bengio and P. Haffner: Gradient-Based Learning Applied to Document Recognition, Proceedings of the IEEE 1998

