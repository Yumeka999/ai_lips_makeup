## -*- coding: utf-8 -*-

import tensorflow as tf # 导入tensorflow处理包
import os
import cv2
from PIL import Image
import numpy as np
import gc
import datetime
from dlib_makeup import Makeup



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # 变量的初始值为截断正太分布
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """
    tf.nn.conv2d功能：给定4维的input和filter，计算出一个2维的卷积结果
    前几个参数分别是input, filter, strides, padding, use_cudnn_on_gpu, ...
    input   的格式要求为一个张量，[batch, in_height, in_width, in_channels],批次数，图像高度，图像宽度，通道数
    filter  的格式为[filter_height, filter_width, in_channels, out_channels]，滤波器高度，宽度，输入通道数，输出通道数
    strides 一个长为4的list. 表示每次卷积以后在input中滑动的距离
    padding 有SAME和VALID两种选项，表示是否要保留不完全卷积的部分。如果是SAME，则保留
    use_cudnn_on_gpu 是否使用cudnn加速。默认是True
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
   

'''
学习参数的配置
'''
image_s = 257                    # 标准的尺寸
mouth_w = 90                     # 嘴巴宽
mouth_h = 30                     # 嘴巴高
chanel = 3                       # RGB 三通道


train_input_fold     = 'data_train/train_month'              # 训练样本输入文件夹
train_output_fold    = 'data_train/train_brightening_month'  # 训练样本输出文件夹
model_fold           = 'model/model_brightening'       # 模型参数文件夹
iteration_record_txt = 'log/record_brightening.txt'    # 保存训练到第几组的参数的记录
log_txt              = 'log/log_brightening.txt'       # 保存每一次迭代的数据

batch_num = 10                     # 读取样本的个数
MAX_INTERATIOIN = 100              # 最大迭代次数设置
checkpoint_steps = 300             # 每迭代50次保存一次模型参数

k_num1 =  24                 # 第一层卷积的个数
k_num2 =  45                 # 第二层卷积的个数
k_s = 3                      # 卷积核的大小

max_num = 1000               # 每个组最大的容量
group_id = 1                 # 读取第几组id的编号
   

'''
group_id 和训练样本编号的联系
group_id = 1 训练样本的编号:0*max_num + 1 ~ 1*max_num
group_id = 2 训练样本的编号:1*max_num + 1 ~ 2*max_num
group_id = 3 训练样本的编号:2*max_num + 1 ~ 3*max_num
group_id = 4 训练样本的编号:3*max_num + 1 ~ 4*max_num
group_id = 5 训练样本的编号:4*max_num + 1 ~ 5*max_num
'''

print('group_id:' + str(group_id) + ' Training number is:' + str((group_id - 1)*max_num) + '~' + str(group_id*max_num))
#样本数据输入
input_image_matrix = [] # 保存输入样本的数值矩阵
num = 0

for f in os.listdir(train_input_fold): # 遍历输入样本文件夹每一个文件
    if num>= (group_id - 1) * max_num and num < group_id*max_num:
        img_month_url = os.path.join(train_input_fold, f)
        image_matrix = cv2.imread(img_month_url)
        image_matrix = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2RGB)
        input_image_matrix.append(image_matrix)
        if num % 10 == 0:
            print('Input ' + str(num) + ' ' + img_month_url)
    elif num > group_id * max_num:
        break
    num = num + 1

# 载入的结束时间
end_time = datetime.datetime.now()
print('group_id:' + str(group_id) + end_time.strftime('%Y-%m-%d %H:%M:%S'))          
print('group_id:' + str(group_id) + ' 输入图片已经载入\n')    

output_image_matrix = []
output_name_list = []

num = 0
for f in os.listdir(train_output_fold):
    if num>= (group_id - 1)*max_num and num < group_id*max_num:
        output_name_list.append(f) # 保存输出文件的名字
        path = os.path.join(train_output_fold, f)
        image_matrix = cv2.imread(path)
        image_matrix = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2RGB)
        output_image_matrix.append(image_matrix)
        num%10 == 0 and print('Output ' + str(num) + ' '  + f)
    elif num > group_id*max_num:
        break
    num = num + 1

# 载入的结束时间
end_time = datetime.datetime.now()
print('group_id:' + str(group_id) + end_time.strftime('%Y-%m-%d %H:%M:%S'))          
print('group_id:' + str(group_id) + ' 输出图片已经载入\n')   

N = len(input_image_matrix) # 训练样本的个数
train_x = input_image_matrix[0:N] # 训练样本输入
valid_x = input_image_matrix[N:]  # 验证样本输入
train_y = output_image_matrix[0:N] # 训练样本输出
valid_y = output_image_matrix[N:] # 验证样本输出
    

# 删除临时变量并立即释放内存
del input_image_matrix
del output_image_matrix
gc.collect()


batch_size = len(train_x)//batch_num # 每个batch样本个数
print('batch_size:' + str(batch_size))
batch_x = [[] for i in range(batch_num)] # 初始化batch的输入
batch_y = [[] for i in range(batch_num)] # 初始化batch的输出

for i in range(batch_num): # 遍历每一个batch
    if i < batch_num - 1:
        for j in range(i*batch_size, (i+1)*batch_size):
            batch_x[i].append(train_x[j])
            batch_y[i].append(train_y[j])     
    else:
        for j in range(i*batch_size, len(train_x)):
            batch_x[i].append(train_x[j])
            batch_y[i].append(train_y[j])
   
        
print('Training Batch finish')

# 删除临时变量并立即释放内存
del train_x
del train_y
gc.collect()
print('内存清理完成')


#网络设计#
#输入图像数据占位符
x = tf.placeholder(tf.float32,  [None, mouth_h, mouth_w, chanel], name='x')  # 输入数据的占位符
y_ = tf.placeholder(tf.float32, [None, mouth_h, mouth_w, chanel], name='y_') # 输出数据的占位符
keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # 这里使用了drop out,即随机安排一些cell输出值为0，可以防止过拟合

"""
# 第一层
# 卷积核(filter)的尺寸是k_s*k_s, 输入通道数为chanel，输出通道为k_num1，即feature map 数目为k_num1
# 又因为strides=[1,1,1,1] 所以单个通道的输出尺寸应该跟输入图像一样。即总的卷积输出应该为?*image_s*image_s*k_num1
# 也就是单个通道输出为image_s*image_s，共有k_num1个通道
"""
W_conv1 = weight_variable([k_s, k_s, chanel, k_num1])  # 卷积是在每个5*5的patch中算出32个特征，分别是patch大小，输入通道数目，输出通道数目
b_conv1 = bias_variable([k_num1])
h_conv1 = tf.nn.elu(conv2d(x, W_conv1) + b_conv1)


"""
# 第二层
# 卷积核k_s*k_s，输入通道为k_num1，输出通道为k_num2。
# 卷积前图像的尺寸为 image_s*image_s*k_num1， 卷积后为 image_s*image_s*k_num2
"""
W_conv2 = weight_variable([k_s, k_s, k_num1, k_num2])
b_conv2 = bias_variable([k_num2])
h_conv2 = tf.nn.elu(conv2d(h_conv1, W_conv2) + b_conv2)

"""
# 第三层
# 卷积核k_s*k_s，输入通道为k_num2，输出通道为chanel。
# 卷积前图像的尺寸为 image_s*image_s*k_num2， 卷积后为 image_s*image_s*chanel
"""
W_fc1 = weight_variable([k_s, k_s, k_num2, chanel])
b_fc1 = bias_variable([chanel])
h_fc1 = tf.nn.elu(conv2d(h_conv2, W_fc1) + b_fc1)

# 损失函数的定义，均方差损失函数
loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.square(h_fc1- y_),reduction_indices=3),reduction_indices=2),reduction_indices=1),reduction_indices=0)


# 使用adam优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver() # 读取tf保存的网络参数
    if os.path.exists(model_fold + '/model.ckpt.index'):
        saver.restore(sess, model_fold + '/model.ckpt')
        print('载入模型参数成功！')
    else:
        print('没有模型参数！')
        pass


    # 程序执行之前清空日志
    with open(log_txt, 'w', buffering= 4*1024) as fp:
        fp.write('')
        fp.flush()


    for i in range(MAX_INTERATIOIN):
        # 迭代开始的时间
        start_time = datetime.datetime.now()

        # 保存每一次迭代的记录
        log_list = []

        # 每一次迭代训练不同的batch
        for j in range(batch_num):
            sess.run(train_step,feed_dict={x: batch_x[j], y_: batch_y[j], keep_prob: 0.5}) # 训练
            lossval = sess.run(loss,feed_dict={x: batch_x[j], y_: batch_y[j], keep_prob: 0.5}) # 损失值
            log_itr_batch = 'step:' + str(i) + ' batch-th:' +str(j) + ' Loss value:' + str(lossval)
            log_list.append(log_itr_batch)
            print(log_itr_batch)

        # 迭代的结束时间
        end_time = datetime.datetime.now()
        print(end_time.strftime('%Y-%m-%d %H:%M:%S'))
        print(str(i) + '-th iteration, Time cost: ' + str((end_time-start_time).seconds))
        print()

        # 每迭代一次写入日志一次
        with open(log_txt, 'a', buffering= 4*1024) as fp:
            for log in log_list:
                fp.write(log+'\n')
            fp.write(end_time.strftime('%Y-%m-%d %H:%M:%S')+'\n')
            fp.write(str(i) + '-th iteration, Time cost: ' + str((end_time-start_time).seconds)+'\n\n')
            fp.flush()

        if (i + 1) % checkpoint_steps == 0: # 每隔50次迭代保存一次模型参数
            saver.save(sess, model_fold + '/model.ckpt')
            # 写入文件表明当前迭代次数
            with open(iteration_record_txt, 'w', buffering= 4*1024) as fp:
                fp.write(str(i + 1)+'\n')
                fp.flush()
                print('迭代次数已经写入' + iteration_record_txt)
    print('网络训练完成!')
    saver = tf.train.Saver() # tf保存网络参数的对象
    saver.save(sess, model_fold + '/model.ckpt')