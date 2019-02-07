import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.io as scio
import csv

#批标准化
def batch_norm(in_image,epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
        return tf.contrib.layers.batch_norm(in_image, decay=momentum, 
                                            updates_collections=None, epsilon=epsilon,
                                            scale=True, scope=name)
        
#卷积
def conv2d(in_image, out_dim, w_row=3, w_col=3, strides_row=2, strides_col=2, stddev=0.02, 
           pad='SAME',name="conv2d"):
    with tf.variable_scope(name):
        w    = tf.get_variable('w', [w_row, w_col, in_image.get_shape()[-1], out_dim], 
                                     initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(in_image, w, strides=[1, strides_row, strides_col, 1], padding=pad)
        b    = tf.get_variable('b', [out_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())
        return conv
    
#激活函数
def swish(x,name='swish'):
    return x*tf.nn.sigmoid(x)

#主体网络内部结构
def dense_block(x_input,conv_dim=16,name='block'):
     with tf.variable_scope(name):
        block1 = conv2d(swish(batch_norm(x_input,name='block1_norm')),conv_dim,
                              w_row=5, w_col=5,strides_row=2,strides_col=2,name='block1')
        #shape=(1, 128, 128, 16) dtype=float32>
        block2 = conv2d(swish(batch_norm(block1,name='block2_norm')),conv_dim,
                              w_row=3, w_col=3,strides_row=1,strides_col=1,name='block2')
        #shape=(1, 128, 128, 16) dtype=float32>
        block3_cat = tf.concat([block1,block2], 3)
        #shape=(1, 128, 128, 32) dtype=float32>
        block3 = conv2d(swish(batch_norm(block3_cat,name='block3_norm')),conv_dim,
                              w_row=3, w_col=3,strides_row=1,strides_col=1,name='block3')
        #shape=(1, 128, 128, 16) dtype=float32>
        block4_cat = tf.concat([block1,block2,block3], 3)
        #shape=(1, 128, 128, 48) dtype=float32>
        block4 = conv2d(swish(batch_norm(block4_cat,name='block4_norm')),conv_dim,
                              w_row=3, w_col=3,strides_row=1,strides_col=1,name='block4')
        #shape=(1, 128, 128, 16) dtype=float32>
        block5_cat = tf.concat([block1,block2,block3,block4], 3)
        #shape=(1, 128, 128, 64) dtype=float32>
        block5 = conv2d(swish(batch_norm(block5_cat,name='block5_norm')),conv_dim,
                              w_row=3, w_col=3,strides_row=1,strides_col=1,name='block5')
        #shape=(1, 128, 128, 16) dtype=float32>
        return tf.concat([block1,block2,block3,block4,block5], 3)
        #shape=(1, 128, 128, 80) dtype=float32>

#主体网络
def densenet(x_input,conv_dim=16,name='densenet'):
    with tf.variable_scope(name):
        conv1 = conv2d(x_input, conv_dim*4, w_row=5, w_col=5,strides_row=2,strides_col=2,name='conv1')
        #shape=(1, 256, 256, 64) dtype=float32>
        conv2 = dense_block(conv1,conv_dim=16,name='conv2')
        #shape=(1, 128, 128, 80) dtype=float32>
        conv3 = conv2d(swish(batch_norm(conv2,name='conv3_norm')),conv_dim*4,
                              w_row=5, w_col=5,strides_row=2,strides_col=2,name='conv3')
        #shape=(1, 64, 64, 64) dtype=float32>
        conv4 = dense_block(conv3,conv_dim=16,name='conv4')
        #shape=(1, 32, 32, 80) dtype=float32>
        conv5 = conv2d(swish(batch_norm(conv4,name='conv5_norm')),conv_dim*4,
                              w_row=5, w_col=5,strides_row=2,strides_col=2,name='conv5')
        #shape=(1, 16, 16, 64) dtype=float32>
        conv6 = dense_block(conv5,conv_dim=16,name='conv6')
        #shape=(1, 8, 8, 80) dtype=float32>
        conv7 = conv2d(swish(batch_norm(conv6,name='conv7_norm')),conv_dim*4,
                              w_row=5, w_col=5,strides_row=2,strides_col=2,name='conv7')
        #shape=(1, 4, 4, 64) dtype=float32>
        conv8 = dense_block(conv7,conv_dim=16,name='conv8')
        #shape=(1, 2, 2, 80) dtype=float32>
        conv9 = conv2d(swish(batch_norm(conv8,name='conv9_norm')),1,
                              w_row=5, w_col=5,strides_row=2,strides_col=2,name='conv9')
        #shape=(1, 1, 1, 1) dtype=float32>
        output = tf.reshape(tf.nn.tanh(conv9),[1])
        #shape=(1,) dtype=float32>
        return output
    
def test_train(patient_number):
    list_dcm = os.listdir(root_path+'\\'+list_patient[patient_number])
    for img_number in range(0,len(list_dcm)):
        data = pydicom.read_file(root_path+'\\'+list_patient[patient_number]+'\\'+list_dcm[img_number])
        img = data.pixel_array
        pre = sess.run(output,feed_dict={x_tem:img,})
        print(pre)
#数据读入
root_path = 'E:\\肝癌影像智慧处理文件夹\\train_dataset'
list_patient = os.listdir(root_path)
#list_patient[0] 0013EDC2-8D7A-4A41-AEB5-D3BB592306D2 存储病人数据的文件夹
label_path = 'E:\\肝癌影像智慧处理文件夹\\train_label.csv'
list_label = csv.reader(open(label_path,'r'))
label_data = []#最终标签就在这里label_data[1][1]
for line in list_label:
    label_data.append(line)

#变量声明
x_tem = tf.placeholder(tf.float32,[512,512])#把病人的一张图片输入到这里
x_input = tf.reshape(x_tem,[1,512,512,1])#塑形,这是输入网络的
label = tf.placeholder(tf.float32)#标签，从csv文件里读入
output = densenet(x_input)#网络的输出
loss = tf.reduce_mean(tf.abs(label-output))#损失函数
train_step = tf.train.AdamOptimizer(0.0002,beta1=0.5).minimize(loss)#优化器

#变量初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#模型训练
'''
patient_number : 表示第几个病人
img_number : 表示当前处理病人的第几张图片for img_number in range(0,len(list_dcm)):
label_data : 第一行是属性，无意义
itera : 迭代次数
每次输入一个病人的一张照片，三重循环，第一重是迭代次数，第二重是3600个病人，第三重是每个病人的所有图片
每个病人对应一个标签
'''
for itera in range(0,15):#0-14
    for patient_number in range(0,3600):#0-3599
        list_dcm = os.listdir(root_path+'\\'+list_patient[patient_number])
        #文件夹里的各个数据的名称 list_dcm[0]就是0013病人的第一张dcm图片
        for img_number in range(0,len(list_dcm)):
            data = pydicom.read_file(root_path+'\\'+list_patient[patient_number]+'\\'+list_dcm[img_number])
            img = data.pixel_array
            name_tem = str(itera)+'/'+str(patient_number)+'/'+str(img_number)
            sess.run(train_step,
                 feed_dict={x_tem:img,label:label_data[patient_number+1][1]})
            #print(name_tem+' end!')
    test_train(0)
'''

'''









