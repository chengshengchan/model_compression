'''
Teacher-Student model implementation by Cheng-Sheng Chan

github : https://github.com/chengshengchan/model_compression
'''

import tensorflow as tf
import numpy as np
import pdb, os, sys
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='teacher-student model')
    parser.add_argument('--model', dest='model', help="model_path to save the student model\n In testing, give trained student model.", type=str)
    parser.add_argument('--task', dest='task', help='task for this file, train/test/val', type=str)
    parser.add_argument('--lr', dest='lr', default=1e-3, help='learning rate', type=float)
    parser.add_argument('--epoch', dest='epoch', default=100, help='total epoch', type=int)
    parser.add_argument('--dropout', dest='dropout', default=0.5, help="dropout ratio", type=float)
    parser.add_argument('--noisy', action='store_true', help='add noisy to logits (noisy-teacher model')
    parser.add_argument('--noisy_ratio', dest='Nratio', default=0.5, help="noisy ratio", type=float)
    parser.add_argument('--noisy_sigma', dest='Nsigma', default=0.9, help="noisy sigma", type=float)
    parser.add_argument('--KD', action='store_true', help='knowledge distilling, hinton 2014')
    parser.add_argument('--lamda', dest='lamda', default=0.3, help='KD method. lamda between original loss and soft-target loss.', type=float)
    parser.add_argument('--tau', dest='tau', default=3.0, help='KD method. tau stands for temperature.', type=float)
    parser.add_argument('--batchsize', dest='batchsize', default=256, type=int)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args, parser

global args, parser
args, parser = parse_args()

class bcolors:
    END  = '\033[0m'  # white (normal)
    R  = '\033[31m' # red
    G  = '\033[32m' # green
    O  = '\033[33m' # orange
    B  = '\033[34m' # blue
    P  = '\033[35m' # purple
    BOLD = '\033[1m'


# function of CNN model reference: https://github.com/aymericdamien/TensorFlow-Examples/
# Create some wrappers for simplicity
def conv(x, W, b, strides=1, padding='SAME'):
    # Conv2D wrapper, with bias and relu activation
    #x = tf.pad(x, paddings, "CONSTANT")
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return x

def maxpool2d(x, k, s, padding='SAME'):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],
                          padding=padding)

def avgpool2d(x, k, s, padding='SAME'):
    # AvgPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],
                          padding=padding)



batch_size=args.batchsize
dim=32
n_classes=10
# placeholders
global x,y,keep_prob

# Create model
def nin(): # modify from model - network in network

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

    # inner product
    ip1 = tf.reshape(cccp5_relu, [-1, weights['ip1'].get_shape().as_list()[0]])
    ip1 = tf.add(tf.matmul(ip1, weights['ip1']), biases['ip1'])
    ip1_relu = tf.nn.relu(ip1)
    ip2 = tf.add(tf.matmul(ip1_relu, weights['ip2']), biases['ip2'])

    return ip2

def lenet(use_pretrained=False): # modify from lenet model

    if use_pretrained == False:
        # Random initialize
        weights = {
            'conv1': tf.get_variable('LN_conv1_w', [5,5,3,64],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
            'conv2': tf.get_variable('LN_conv2_w', [5,5,64,128],initializer=tf.contrib.layers.xavier_initializer_conv2d()),
            'ip1': tf.get_variable('LN_ip1_w', [5*5*128, 1024] , initializer=tf.contrib.layers.xavier_initializer()),
            'ip2': tf.get_variable('LN_ip2_w', [1024,10], initializer=tf.contrib.layers.xavier_initializer())
        }

        biases = {
            'conv1': tf.Variable(tf.random_normal(shape=[64],stddev=0.5), name = 'LN_conv1_b'),
            'conv2': tf.Variable(tf.random_normal(shape=[128],stddev=0.5), name = 'LN_conv2_b'),
            'ip1': tf.Variable(tf.random_normal(shape=[1024],stddev=0.5), name = 'LN_ip1_b'),
            'ip2': tf.Variable(tf.random_normal(shape=[10],stddev=0.5), name = 'LN_ip2_b')
        }
    else:
        # initialized by pre-trained weight
        npyfile = np.load('student.npy')
        npyfile = npyfile.item()
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

    conv1 = conv(x, weights['conv1'], biases['conv1'],padding='VALID')
    pool1 = maxpool2d(conv1, k=2, s=2)
    conv2 = conv(pool1, weights['conv2'], biases['conv2'], padding='VALID')
    pool2 = maxpool2d(conv2, k=2, s=2,padding='VALID')

    ip1 = tf.reshape(pool2, [-1, weights['ip1'].get_shape().as_list()[0]])
    ip1 = tf.add(tf.matmul(ip1, weights['ip1']), biases['ip1'])
    ip1_relu = tf.nn.relu(ip1)
    ip2 = tf.add(tf.matmul(ip1_relu, weights['ip2']), biases['ip2'])
    return ip2

# Reading cifar dataset
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
        data = np.uint8(np.zeros((50000,32,32,3)))
        label=np.int16(np.zeros((50000,1)))
    elif flag == 'test':
        batches = ['test_batch']
        data = np.float32(np.zeros((10000,32,32,3)))
        label= np.zeros((10000,1))

    N = 10000
    for i in xrange(len(batches)):
        b = batches[i]
        temp = unpickle(os.path.join(path,b))
        for j in xrange(N):
            data[N*i+j][:,:,2] = np.reshape(temp['data'][j][2048:],(32,32))
            data[N*i+j][:,:,1] = np.reshape(temp['data'][j][1024:2048],(32,32))
            data[N*i+j][:,:,0] = np.reshape(temp['data'][j][:1024],(32,32))
            label[N*i+j] = temp['labels'][j]
    return data, label


def train():
    learning_rate=args.lr
    model_path=args.model
    total_epoch = args.epoch
    teacher=nin()
    student=lenet()
    if args.noisy == True:
        drop_scale = 1/args.Nratio
        noisy_mask = tf.nn.dropout( tf.constant(np.float32(np.ones((batch_size,1)))/drop_scale) ,keep_prob=args.Nratio) #(batchsize,1)
        gaussian = tf.random_normal(shape=[batch_size,1], mean=0.0, stddev=args.Nsigma)
        noisy = tf.mul(noisy_mask, gaussian)
        #noisy_add = tf.add(tf.constant(np.float32(np.ones((batch_size,1)))), noisy)
        teacher = tf.mul(teacher, tf.tile(noisy,tf.constant([1,10])))   #(batchsize,10)
        #teacher = tf.add(teacher, tf.tile(noisy,tf.constant([1,10])))
        print bcolors.G+"prepare for training, noisy mode"+bcolors.END
        tf_loss = tf.nn.l2_loss(teacher - student)/batch_size
    elif args.KD == True:   # correct Hinton method at 2017.1.3
        print bcolors.G+"prepare for training, knowledge distilling mode"+bcolors.END
        one_hot = tf.one_hot(y, n_classes,1.0,0.0)
        #one_hot = tf.cast(one_hot_int, tf.float32)
        teacher_tau = tf.scalar_mul(1.0/args.tau, teacher)
        student_tau = tf.scalar_mul(1.0/args.tau, student)
        objective1 = tf.nn.sigmoid_cross_entropy_with_logits(student_tau, one_hot)
        objective2 = tf.scalar_mul(0.5, tf.square(student_tau-teacher_tau))
        tf_loss = (args.lamda*tf.reduce_sum(objective1) + (1-args.lamda)*tf.reduce_sum(objective2))/batch_size
    else:
        print bcolors.G+"prepare for training, NIPS2014 mode"+bcolors.END
        tf_loss = tf.nn.l2_loss(teacher - student)/batch_size

    optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(tf_loss)
    optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate/10).minimize(tf_loss)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
    tf.initialize_all_variables().run()
    with tf.device('/cpu:0'):
        saver = tf.train.Saver(max_to_keep=100)
        #saver.restore(sess, os.path.join(model_path,'model-99')
    data, label=read_cifar10('train')
    index=np.array(range(len(data)))    # index randomly ordered
    mean = cal_mean()
    begin = time.time()
    iterations = len(data)/batch_size
    decay_step = int(total_epoch*0.8)
    cnt=0
    dropout_rate=args.dropout
    print bcolors.G+"number of iterations (per epoch) ="+str(len(data)/batch_size)+bcolors.END
    for i in xrange(total_epoch):
        np.random.shuffle(index)
        cost_sum=0
        for j in xrange(iterations):
            batch_x = np.float32(data[index[j*batch_size:(j+1)*batch_size]]) - mean
            batch_y = np.squeeze(np.float32(label[index[j*batch_size:(j+1)*batch_size]]))
            if cnt/decay_step == 0:
                lr=learning_rate
                _, cost = sess.run([optimizer1, tf_loss],
                    feed_dict={x : batch_x, y : batch_y, keep_prob : 1-dropout_rate})
            elif cnt/decay_step == 1:
                lr=learning_rate/10
                _, cost = sess.run([optimizer2, tf_loss],
                    feed_dict={x : batch_x, y : batch_y, keep_prob : 1-dropout_rate})
            cost_sum += cost
            #pdb.set_trace()
            #if (j % int(iterations*0.25) == 0):
            #    print ("epoch %d-iter %d, cost = %f , avg-cost = %f"%(i, j, cost, cost/n_classes))
            #    sys.stdout.flush()
        cnt +=1
        avg_time = time.time()-begin
        print ("epoch %d - avg. %f seconds in each epoch, lr = %.0e, cost = %f , avg-cost-per-logits = %f"%(i, avg_time/cnt,lr, cost_sum, cost_sum/iterations/n_classes))
        if np.mod(i+1, 10) == 0:
            print ("Epoch ", i+1, " is done. Saving the model ...")
            with tf.device('/cpu:0'):
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                saver.save(sess, os.path.join(model_path, 'model'), global_step=i)
        sys.stdout.flush()


def test():
    print bcolors.G+"Task : test\n"+bcolors.END
    student = lenet()
    pred = tf.nn.softmax(student)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    init=tf.initialize_all_variables()
    sess.run(init)
    with tf.device('/cpu:0'):
        saver = tf.train.Saver()
        saver.restore(sess, args.model)

    mean = cal_mean()
    data, label=read_cifar10('test')
    total=0
    correct=0
    begin = time.time()
    for j in xrange(len(data)/batch_size):
        batch_x = data[j*batch_size:(j+1)*batch_size] - mean
        prob = sess.run([pred],
                feed_dict={x : batch_x, y : np.ones((batch_size)), keep_prob : 1.0})
        if np.argmax(prob[0]) == label[j]:
            correct += 1
        total+=1
        #print ("acc = %f . %d/%d"%(float(correct)/total, correct, total))
    end = time.time()
    print ("acc = %f . %d/%d.  Computing time = %f seconds"%(float(correct)/total, correct, total, end-begin))


def valid_nin():
    print bcolors.G+"Task : val\nvalidate the pre-trained nin model, should be same as caffe result"+bcolors.END
    pool3=nin()
    #pool3 = lenet()
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
                feed_dict={x : batch_x, y : np.ones((batch_size)), keep_prob : 1.0})
        if np.argmax(prob[0]) == label[j]:
            correct += 1
        total+=1
    end = time.time()
    print ("acc = %f . %d/%d.  Computing time = %f seconds"%(float(correct)/total, correct, total, end-begin))


if __name__ == '__main__':
    global batch_size
    print bcolors.G+"Reading args...."
    print args
    print bcolors.END
    if args.noisy == True and args.KD == True:
        print bcolors.BOLD+bcolors.R+"Invalid args!\n"+bcolors.END+bcolors.R+"only one method can be selected, noisy or KD(knowledge distilling)"+bcolors.END
        exit(1)
    if args.task=='test' or args.task=='val':
        batch_size=1

    x = tf.placeholder(tf.float32, [batch_size, dim, dim, 3])
    y = tf.placeholder(tf.int32, [batch_size])
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
    with tf.device('/gpu:0'):
        if args.task=='train':
            train()
        elif args.task=='test':
            test()
        elif args.task=='val':
            valid_nin()
        else:
            print bcolors.BOLD+bcolors.R+"Invalid args!\n"+bcolors.END+bcolors.R+"task should be train, test, or val"+bcolors.END
