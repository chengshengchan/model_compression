'''
Teacher-Student model implementation
CNN model ref: Project: https://github.com/aymericdamien/TensorFlow-Examples/
conver caffe model weights to npy : https://github.com/ethereon/caffe-tensorflow

Dataset : cifar-10
'''




#from __future__ import print_function

import tensorflow as tf
import numpy as np
import pdb, os, sys
import time

# Create some wrappers for simplicity
def conv(x, W, b, strides=1, padding='SAME'):
    # Conv2D wrapper, with bias and relu activation
    #x = tf.pad(x, paddings, "CONSTANT")
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return x
'''

#def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
def conv(inputX, kernel, biases, strides=1, padding='SAME', group=1):
    #From https://github.com/ethereon/caffe-tensorflow
    #        http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
    #        k_h, k_w : kernel height/width
    #        c_o : channel_output
    #        s_h, s_w : stride height/width
    
    k_h = int(kernel.get_shape()[0])
    k_w = int(kernel.get_shape()[1])
    c_o = int(kernel.get_shape()[3])
    s_h = strides
    s_w = strides

    c_i = inputX.get_shape()[-1]
    #inputX = tf.pad(inputX, padding, "CONSTANT")

    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    if group==1:
        conv = convolve(inputX, kernel)
    else:
        input_groups = tf.split(3, group, inputX)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])
'''

def maxpool2d(x, k, s, padding='SAME'):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],
                          padding=padding)

def avgpool2d(x, k, s, padding='SAME'):
    # AvgPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],
                          padding=padding)

batch_size=256
#batch_size=1
dim=32
n_classes=10
# placeholder
x = tf.placeholder(tf.float32, [batch_size, dim, dim, 3])
y = tf.placeholder(tf.float32, [batch_size, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create model
def nin_net(): # modify from model - network in network

    # pre-trained weight
    npyfile = np.load('teacher.npy')
    npyfile = npyfile.item()

    weights = {
        'conv1': tf.Variable(npyfile['conv1']['weights'], trainable=False, name = 'conv1_w'),
        'cccp1': tf.Variable(npyfile['cccp1']['weights'], trainable=False, name = 'cccp1_w'),
        'cccp2': tf.Variable(npyfile['cccp2']['weights'], trainable=False, name = 'cccp2_w'),
        'conv2': tf.Variable(npyfile['conv2']['weights'], trainable=False, name = 'conv2_w'),
        'cccp3': tf.Variable(npyfile['cccp3']['weights'], trainable=False, name = 'cccp3_w'),
        'cccp4': tf.Variable(npyfile['cccp4']['weights'], trainable=False, name = 'cccp4_w'),
        'conv3': tf.Variable(npyfile['conv3']['weights'], trainable=False, name = 'conv3_w'),
        'cccp5': tf.Variable(npyfile['cccp5']['weights'], trainable=False, name = 'cccp5_w'),
        'ip1': tf.Variable(npyfile['ip1']['weights'], trainable=False, name = 'ip1_w'),
        'ip2': tf.Variable(npyfile['ip2']['weights'], trainable=False, name = 'ip2_w')
    }

    biases = {
        'conv1': tf.Variable(npyfile['conv1']['biases'], trainable=False, name = 'conv1_b'),
        'cccp1': tf.Variable(npyfile['cccp1']['biases'], trainable=False, name = 'cccp1_b'),
        'cccp2': tf.Variable(npyfile['cccp2']['biases'], trainable=False, name = 'cccp2_b'),
        'conv2': tf.Variable(npyfile['conv2']['biases'], trainable=False, name = 'conv2_b'),
        'cccp3': tf.Variable(npyfile['cccp3']['biases'], trainable=False, name = 'cccp3_b'),
        'cccp4': tf.Variable(npyfile['cccp4']['biases'], trainable=False, name = 'cccp4_b'),
        'conv3': tf.Variable(npyfile['conv3']['biases'], trainable=False, name = 'conv3_b'),
        'cccp5': tf.Variable(npyfile['cccp5']['biases'], trainable=False, name = 'cccp5_b'),
        'ip1': tf.Variable(npyfile['ip1']['biases'], trainable=False, name = 'ip1_b'),
        'ip2': tf.Variable(npyfile['ip2']['biases'], trainable=False, name = 'ip2_b')
    }

    conv1 = conv(x, weights['conv1'], biases['conv1'])
    conv1_relu = tf.nn.relu(conv1)
    cccp1 = conv(conv1_relu, weights['cccp1'], biases['cccp1'])
    cccp1_relu = tf.nn.relu(cccp1)
    cccp2 = conv(cccp1_relu, weights['cccp2'], biases['cccp2'])
    cccp2_relu = tf.nn.relu(cccp2)
    pool1 = maxpool2d(cccp2_relu, k=3, s=2)
    drop3 = tf.nn.dropout(pool1, keep_prob)

    conv2 = conv(drop3, weights['conv2'], biases['conv2'])
    conv2_relu = tf.nn.relu(conv2)
    cccp3 = conv(conv2_relu, weights['cccp3'], biases['cccp3'])
    cccp3_relu = tf.nn.relu(cccp3)
    cccp4 = conv(cccp3_relu, weights['cccp4'], biases['cccp4'])
    cccp4_relu = tf.nn.relu(cccp4)

    pool2 = avgpool2d(cccp4_relu, k=3, s=2)
    drop6 = tf.nn.dropout(pool2, keep_prob)
    
    conv3 = conv(drop6, weights['conv3'], biases['conv3'])
    conv3_relu = tf.nn.relu(conv3)
    cccp5 = conv(conv3_relu, weights['cccp5'], biases['cccp5'])
    cccp5_relu = tf.nn.relu(cccp5)
    
    ip1 = tf.reshape(cccp5_relu, [-1, weights['ip1'].get_shape().as_list()[0]])
    ip1 = tf.add(tf.matmul(ip1, weights['ip1']), biases['ip1'])
    ip1_relu = tf.nn.relu(ip1)
    ip2 = tf.add(tf.matmul(ip1_relu, weights['ip2']), biases['ip2'])
    
    #ip2 = tf.nn.softmax(ip2)
    return ip2

def lenet_modify(): # modify from lenet model
    # pre-trained weight
    npyfile = np.load('student.npy')
    npyfile = npyfile.item()
    dev=0.01
    weights = {
        'conv1': tf.get_variable('LN_conv1_w', [5,5,3,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'conv2': tf.get_variable('LN_conv2_w', [5,5,64,128],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'ip1': tf.get_variable('LN_ip1_w', [5*5*128, 1024] , initializer=tf.contrib.layers.xavier_initializer()),
        'ip2': tf.get_variable('LN_ip2_w', [1024,10], initializer=tf.contrib.layers.xavier_initializer())
    }

    biases = {
        'conv1': tf.Variable(tf.random_normal(shape=[64],stddev=1.0), trainable=True, name = 'LN_conv1_b'),
        'conv2': tf.Variable(tf.random_normal(shape=[128],stddev=1.0), trainable=True, name = 'LN_conv2_b'),
        'ip1': tf.Variable(tf.random_normal(shape=[1024],stddev=1.0), trainable=True, name = 'LN_ip1_b'),
        'ip2': tf.Variable(tf.random_normal(shape=[10],stddev=1.0), trainable=True, name = 'LN_ip2_b')
    }
    '''
    weights = {
        'conv1': tf.Variable(tf.random_normal(shape=[5,5,3,64],stddev=dev), trainable=True, name = 'LN_conv1_w'),
        'conv2': tf.Variable(tf.random_normal(shape=[5,5,64,128],stddev=dev), trainable=True, name = 'LN_conv2_w'),
        'ip1': tf.Variable(tf.random_normal(shape=[5*5*128,1024],stddev=dev), trainable=True, name = 'LN_ip1_w'),
        'ip2': tf.Variable(tf.random_normal(shape=[1024,10],stddev=dev), trainable=True, name = 'LN_ip2_w')
    }

    biases = {
        'conv1': tf.Variable(tf.random_normal(shape=[64],stddev=dev), trainable=True, name = 'LN_conv1_b'),
        'conv2': tf.Variable(tf.random_normal(shape=[128],stddev=dev), trainable=True, name = 'LN_conv2_b'),
        'ip1': tf.Variable(tf.random_normal(shape=[1024],stddev=dev), trainable=True, name = 'LN_ip1_b'),
        'ip2': tf.Variable(tf.random_normal(shape=[10],stddev=dev), trainable=True, name = 'LN_ip2_b')
    }
    '''
    '''
    weights = {
        'conv1': tf.Variable(npyfile['conv1']['weights'], name = 'LN_conv1_w'),
        'conv2': tf.Variable(npyfile['conv2']['weights'], name = 'LN_conv2_w'),
        'ip1': tf.Variable(npyfile['ip1']['weights'], name = 'LN_ip1_w'),
        'ip2': tf.Variable(npyfile['ip2']['weights'], name = 'LN_ip2_w'),
    }

    biases = {
        'conv1': tf.Variable(npyfile['conv1']['biases'], name = 'LN_conv1_b'),
        'conv2': tf.Variable(npyfile['conv2']['biases'], name = 'LN_conv2_b'),
        'ip1': tf.Variable(npyfile['ip1']['biases'], name = 'LN_ip1_b'),
        'ip2': tf.Variable(npyfile['ip2']['biases'], name = 'LN_ip2_b'),
    }
    '''
    conv1 = conv(x, weights['conv1'], biases['conv1'],padding='VALID')
    pool1 = maxpool2d(conv1, k=2, s=2)
    conv2 = conv(pool1, weights['conv2'], biases['conv2'], padding='VALID')
    pool2 = maxpool2d(conv2, k=2, s=2,padding='VALID')

    ip1 = tf.reshape(pool2, [-1, weights['ip1'].get_shape().as_list()[0]])
    ip1 = tf.add(tf.matmul(ip1, weights['ip1']), biases['ip1'])
    ip1_relu = tf.nn.relu(ip1)
    ip2 = tf.add(tf.matmul(ip1_relu, weights['ip2']), biases['ip2'])
    
    #ip2 = tf.nn.softmax(ip2)
    return ip2, weights


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def cal_mean():
    data, label = read_cifar10('train')
    mean = np.mean(data, axis=0)
    return mean


def read_cifar10(flag):
    # Directory of cifar-10
    path = '/home/james/Desktop/workspace/model_compression/cifar-10-batches-py'
    if flag == 'train':
        batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        data = np.float32(np.zeros((50000,32,32,3)))
        label=np.zeros((50000,1))
    elif flag == 'test':
        batches = ['test_batch']
        data = np.float32(np.zeros((10000,32,32,3)))
        label= np.zeros((10000,1))
    
    for i in xrange(len(batches)):
        b = batches[i]
        temp = unpickle(os.path.join(path,b))
        for j in xrange(10000):
            data[10000*i+j][:,:,2] = np.reshape(temp['data'][j][2048:],(32,32))
            data[10000*i+j][:,:,1] = np.reshape(temp['data'][j][1024:2048],(32,32))
            data[10000*i+j][:,:,0] = np.reshape(temp['data'][j][:1024],(32,32))
            label[10000*i+j] = temp['labels'][j]
    return data, label


def train():
    learning_rate=1e-7
    model_path='nips2014'
    total_epoch = 2000
    teacher=nin_net()
    student, weights=lenet_modify()
    #tf_loss = tf.nn.l2_loss(tf.log(tf.clip_by_value(teacher,1e-10,1))-tf.log(tf.clip_by_value(student,1e-10,1)))/batch_size
    tf_loss = tf.nn.l2_loss(teacher - student)/batch_size

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

    optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(tf_loss)
    optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate/10).minimize(tf_loss)
    optimizer3 = tf.train.AdamOptimizer(learning_rate=learning_rate/100).minimize(tf_loss)
    optimizer4 = tf.train.AdamOptimizer(learning_rate=learning_rate/1000).minimize(tf_loss)
    optimizer5 = tf.train.AdamOptimizer(learning_rate=learning_rate/10000).minimize(tf_loss)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    #init=tf.initialize_all_variables()
    #sess.run(init)
    tf.initialize_all_variables().run()
    with tf.device('/cpu:0'):
        saver = tf.train.Saver(max_to_keep=100)
        saver.restore(sess, 'nips2014/model-999')
    data, _=read_cifar10('train')
    index=np.array(range(len(data)))
    mean = cal_mean()
    begin = time.time()
    iterations = len(data)/batch_size
    decay_step = 300
    cnt=0
    for i in xrange(1000,total_epoch):
        np.random.shuffle(index)
        cost_sum=0
        for j in xrange(len(data)/batch_size):
            batch_x = data[index[j*batch_size:(j+1)*batch_size]] - mean
            if cnt/decay_step == 0:
                lr=learning_rate
                _, cost, T, S = sess.run([optimizer1, tf_loss, teacher, student],
                    feed_dict={x : batch_x, y : np.ones((batch_size, n_classes)), keep_prob : 0.5})
            elif cnt/decay_step == 1:
                lr=learning_rate/10
                _, cost, T, S = sess.run([optimizer2, tf_loss, teacher, student],
                    feed_dict={x : batch_x, y : np.ones((batch_size, n_classes)), keep_prob : 0.5})
            elif cnt/decay_step == 2:
                lr=learning_rate/100
                _, cost, T, S = sess.run([optimizer3, tf_loss, teacher, student],
                    feed_dict={x : batch_x, y : np.ones((batch_size, n_classes)), keep_prob : 0.5})
            elif cnt/decay_step == 3:
                lr=learning_rate/1000
                _, cost, T, S = sess.run([optimizer4, tf_loss, teacher, student],
                    feed_dict={x : batch_x, y : np.ones((batch_size, n_classes)), keep_prob : 0.8})
            elif cnt/decay_step == 4:
                lr=learning_rate/10000
                _, cost, T, S = sess.run([optimizer5, tf_loss, teacher, student],
                    feed_dict={x : batch_x, y : np.ones((batch_size, n_classes)), keep_prob : 0.8})
            cost_sum += cost
            #pdb.set_trace()
            #print ("cost = %f , avg-cost = %f"%(cost, cost/n_classes))
        cnt +=1
        avg_time = time.time()-begin
        print ("epoch %d - avg. %f seconds in each epoch, lr = %.0e, cost = %f , avg-cost = %f"%(i, avg_time/cnt,lr, cost_sum, cost_sum/iterations/n_classes))
        if np.mod(i+1, 100) == 0:
            print ("Epoch ", i, " is done. Saving the model ...")
            with tf.device('/cpu:0'):
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                saver.save(sess, os.path.join(model_path, 'model'), global_step=i)
        sys.stdout.flush()


def test():
    student ,w = lenet_modify()
    pred = tf.nn.softmax(student)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    #init=tf.initialize_all_variables()
    #sess.run(init)
    with tf.device('/cpu:0'):
        saver = tf.train.Saver()
        saver.restore(sess, 'model/model-899')

    mean = cal_mean()
    data, label=read_cifar10('test')
    total=0
    correct=0
    for j in xrange(len(data)/batch_size):
        batch_x = data[j*batch_size:(j+1)*batch_size] - mean
        prob = sess.run([pred],
                feed_dict={x : batch_x, y : np.ones((batch_size, n_classes)), keep_prob : 1.0})
        if np.argmax(prob[0]) == label[j]:
            correct += 1
        total+=1
        print ("acc = %f . %d/%d"%(float(correct)/total, correct, total))

# arXiv 2016.
def train_noisy():
    learning_rate=1e-3
    model_path='noisy'
    total_epoch = 1000
    teacher=nin_net()
    student, weights=lenet_modify()
    #tf_loss = tf.nn.l2_loss(tf.log(tf.clip_by_value(teacher,1e-10,1))-tf.log(tf.clip_by_value(student,1e-10,1)))/batch_size
    drop_scale = 1/0.5
    noisy_mask = tf.nn.dropout( tf.constant(np.float32(np.ones((batch_size,1)))/drop_scale) ,keep_prob=0.5)
    gaussian = tf.random_normal(shape=[batch_size,1], mean=0.0, stddev=0.9)
    noisy = tf.mul(noisy_mask, gaussian)
    #noisy_add = tf.add(tf.constant(np.float32(np.ones((batch_size,1)))), noisy)
    teacher = tf.add(teacher, tf.tile(noisy,tf.constant([1,10])))
    tf_loss = tf.nn.l2_loss(teacher - student)/batch_size

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

    optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(tf_loss)
    optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate/10).minimize(tf_loss)
    optimizer3 = tf.train.AdamOptimizer(learning_rate=learning_rate/100).minimize(tf_loss)
    optimizer4 = tf.train.AdamOptimizer(learning_rate=learning_rate/1000).minimize(tf_loss)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    #init=tf.initialize_all_variables()
    #sess.run(init)
    tf.initialize_all_variables().run()
    with tf.device('/cpu:0'):
        saver = tf.train.Saver(max_to_keep=100)
        saver.restore(sess, 'noisy/model-199')
    data, _=read_cifar10('train')
    index=np.array(range(len(data)))
    mean = cal_mean()
    begin = time.time()
    iterations = len(data)/batch_size
    decay_step = 250

    for i in xrange(200,total_epoch):
        np.random.shuffle(index)
        cost_sum=0
        for j in xrange(len(data)/batch_size):
            batch_x = data[index[j*batch_size:(j+1)*batch_size]] - mean
            if i/decay_step == 0:
                lr=learning_rate*1.0
                _, cost, T, S = sess.run([optimizer1, tf_loss, teacher, student],
                    feed_dict={x : batch_x, y : np.ones((batch_size, n_classes)), keep_prob : 1.0})
            elif i/decay_step == 1:
                lr=learning_rate/10
                _, cost, T, S = sess.run([optimizer2, tf_loss, teacher, student],
                    feed_dict={x : batch_x, y : np.ones((batch_size, n_classes)), keep_prob : 1.0})
            elif i/decay_step == 2:
                lr=learning_rate/100
                _, cost, T, S = sess.run([optimizer3, tf_loss, teacher, student],
                    feed_dict={x : batch_x, y : np.ones((batch_size, n_classes)), keep_prob : 1.0})
            elif i/decay_step == 3:
                lr=learning_rate/1000
                _, cost, T, S = sess.run([optimizer4, tf_loss, teacher, student],
                    feed_dict={x : batch_x, y : np.ones((batch_size, n_classes)), keep_prob : 1.0})
            cost_sum += cost
            #pdb.set_trace()
            #print ("cost = %f , avg-cost = %f"%(cost, cost/n_classes))
        avg_time = time.time()-begin
        print ("epoch %d - avg. %f seconds in each epoch, lr = %.0e, cost = %f , avg-cost = %f"%(i, avg_time/(i+1),lr, cost_sum, cost_sum/iterations/n_classes))
        if np.mod(i+1, 100) == 0:
            print ("Epoch ", i, " is done. Saving the model ...")
            with tf.device('/cpu:0'):
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                saver.save(sess, os.path.join(model_path, 'model'), global_step=i)
        sys.stdout.flush()
def valid_nin():
    #pool3=nin_net()
    pool3,w = lenet_modify()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    init=tf.initialize_all_variables()
    sess.run(init)
    mean = cal_mean()
    data, label=read_cifar10('test')
    total=0
    correct=0
    begin = time.time()
    for j in xrange(len(data)/batch_size):
        batch_x = data[j*batch_size:(j+1)*batch_size] - mean
        prob = sess.run([pool3],
                feed_dict={x : batch_x, y : np.ones((batch_size, n_classes)), keep_prob : 1.0})
        if np.argmax(prob[0]) == label[j]:
            correct += 1
        total+=1
    end = time.time()
    print ("acc = %f . %d/%d.  Computing time = %f seconds"%(float(correct)/total, correct, total, end-begin))


if __name__ == '__main__':
    
    with tf.device('/gpu:0'):
        #train_noisy()
        train()
    #with tf.device('/gpu:0'):
        #pass
        #test()
        #valid_nin()
