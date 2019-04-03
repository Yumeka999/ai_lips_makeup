# -*- coding: utf-8 -*-

import tensorflow as tf # 导入tensorflow处理包
import setting
import os

class AIMakeupPredict:
    def __init__(self):
        self.chanel  = 3
        self.mouth_w = 90  # 嘴巴宽
        self.mouth_h = 30  # 嘴巴高
        self.k_num1  = 24  # 第一层卷积的个数
        self.k_num2  = 45  # 第二层卷积的个数
        self.k_s     = 3  # 卷积核的大小

        # 网络设计#
        # 输入图像数据占位符
        self.x  = tf.placeholder(tf.float32, [None, self.mouth_h, self.mouth_w, self.chanel], name='x')   # 输入数据的占位符
        self.y_ = tf.placeholder(tf.float32, [None, self.mouth_h, self.mouth_w, self.chanel], name='y_')  # 输出数据的占位符
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')                      # 这里使用了drop out,即随机安排一些cell输出值为0，可以防止过拟合

        """
        # 第一层
        # 卷积核(filter)的尺寸是k_s*k_s, 输入通道数为chanel，输出通道为k_num1，即feature map 数目为k_num1
        # 又因为strides=[1,1,1,1] 所以单个通道的输出尺寸应该跟输入图像一样。即总的卷积输出应该为?*image_s*image_s*k_num1
        # 也就是单个通道输出为image_s*image_s，共有k_num1个通道
        """
        self.W_conv1 = self.weight_variable([self.k_s, self.k_s, self.chanel, self.k_num1])  # 卷积是在每个5*5的patch中算出32个特征，分别是patch大小，输入通道数目，输出通道数目
        self.b_conv1 = self.bias_variable([self.k_num1])
        self.h_conv1 = tf.nn.elu(self.conv2d(self.x, self.W_conv1) + self.b_conv1)

        """
        # 第二层
        # 卷积核k_s*k_s，输入通道为k_num1，输出通道为k_num2。
        # 卷积前图像的尺寸为 image_s*image_s*k_num1， 卷积后为 image_s*image_s*k_num2
        """
        self.W_conv2 = self.weight_variable([self.k_s, self.k_s, self.k_num1, self.k_num2])
        self.b_conv2 = self.bias_variable([self.k_num2])
        self.h_conv2 = tf.nn.elu(self.conv2d(self.h_conv1, self.W_conv2) + self.b_conv2)

        """
        # 第三层
        # 卷积核k_s*k_s，输入通道为k_num2，输出通道为chanel。
        # 卷积前图像的尺寸为 image_s*image_s*k_num2， 卷积后为 image_s*image_s*chanel
        """

        self.W_fc1 = self.weight_variable([self.k_s, self.k_s, self.k_num2, self.chanel])
        self.b_fc1 = self.bias_variable([self.chanel])
        self.h_fc1 = tf.nn.elu(self.conv2d(self.h_conv2, self.W_fc1) + self.b_fc1)

        self.brightening_sess  = tf.Session()  # 保存的session
        self.brightening_saver = tf.train.Saver()  # 读取tf保存的网络参数

        if os.path.exists(setting.s_brightening_saved_modl_idx):
            self.brightening_saver.restore(self.brightening_sess, setting.s_brightening_saved_modl)
            print('釉面网络参数载入成功！')
        else:
            print('釉面网络参数文件夹不存在!')
            pass

        self.whitening_sess  = tf.Session()  # 保存的session
        self.whitening_saver = tf.train.Saver()  # 读取tf保存的网络参数

        if os.path.exists(setting.s_whitening_saved_modl_idx):
            self.whitening_saver.restore(self.whitening_sess, setting.s_whitening_saved_modl)
            print('亮面网络参数载入成功！')
        else:
            print('亮面网络参数文件夹不存在!')
            pass


    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1) # 变量的初始值为截断正太分布
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
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

    # 预测釉面口红
    def predict_brightening(self, mouth_matrix, mouth_out):
        p_brightening = self.brightening_sess.run(self.h_fc1, feed_dict={self.x: [mouth_matrix], self.y_: [mouth_out], self.keep_prob: 0.5})
        return p_brightening

    # 预测亮面口红
    def predict_whitening(self, mouth_matrix, mouth_out):
        p_whitening   = self.whitening_sess.run(  self.h_fc1, feed_dict={self.x: [mouth_matrix], self.y_: [mouth_out], self.keep_prob: 0.5})
        return p_whitening


