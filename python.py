# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 00:13:40 2019

@author: ASUS
"""
###环境布置很重要，3.5和2的py支持tensorflow，ps.环境布置就试了好久

###############################################################################
#安装模块和环境，有需要请在cmd命令行中pip install，注意已安装python版本与其所支持的模块版本
import os#os模块调用系统命令
import sys#包含Python解释器和它的环境有关的函数
import numpy as np#使用的numpy模块中的随机函数
import scipy.io
import scipy.misc
import tensorflow as tf  #用于各类机器学习算法，官方安装方法常会遇到bug，具体解决见论文
###############################################################################
# 设置位置与生成图片常规
OUTPUT_DIR = 'images/output-macau/'# 转换出的图片保存路径
STYLE_IMAGE = 'images/StarryNight.jpg'# 样式图片的保存路径
CONTENT_IMAGE = 'images/Macau.jpg'# 要转换的图片的保存路径
# 图像尺寸大小常量
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
COLOR_CHANNELS = 3

###############################################################################
#算法的常量设置  提前下载VGG 19
NOISE_RATIO = 0.6
ITERATIONS = 5000# 迭代次数.
BETA = 5# 定义内容流失数
ALPHA = 100# 定义风格流失数
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'#VGG 19层模型深度学习
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
#在不更改数组数据的情况下为数组提供新形状

###############################################################################
#运用转移学习的方法，通过预先训练过的模型来画画
def generate_noise_image(content_image, noise_ratio = NOISE_RATIO):
   #返回与内容图像以一定比例混合的噪声图像.
    noise_image = np.random.uniform(
            -20, 20,
            (1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)).astype('float32')
    # 取这些值的加权平均值
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    return input_image
def load_image(path):
    image = scipy.misc.imread(path)#读入
    # 重置图像的大小，其余不变
    image = np.reshape(image, ((1,) + image.shape))
    # 减去平均值后输入VGG模型.
    image = image - MEAN_VALUES
    return image
def save_image(path, image):
    image = image + MEAN_VALUES # 输出应将平均值相加.
    image = image[0]# 去掉第一个无用的维度，剩下的是图像.
    image = np.clip(image, 0, 255).astype('uint8')#截取函数，将范围外的数强制转化为范围内的数
    scipy.misc.imsave(path, image)#处理图像存储
def load_vgg_model(path):
    vgg = scipy.io.loadmat(path)#vgg加载处理图像位置
    vgg_layers = vgg['layers']
    def _weights(layer, expected_layer_name):
        #返回给定层的VGG模型的权重和偏差
        W = vgg_layers[0][layer][0][0][0][0][0]
        b = vgg_layers[0][layer][0][0][0][0][1]
        layer_name = vgg_layers[0][layer][0][0][-2]
        assert layer_name == expected_layer_name
        return W, b
    def _relu(conv2d_layer):
        #返回封装在TensorFlow层上的RELU函数
        return tf.nn.relu(conv2d_layer)
    def _conv2d(prev_layer, layer, layer_name):
        #使用权重返回Conv2D层，来自VGG的偏差在“层”模型。
        W, b = _weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.conv2d(
            prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b
    def _conv2d_relu(prev_layer, layer, layer_name):
        #返回Conv2D + RELU层使用的权重，偏差从VGG在“层”模型。
        return _relu(_conv2d(prev_layer, layer, layer_name))
    def _avgpool(prev_layer):
#返回
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
   
###############################################################################
# 构建图模型.
    graph = {}
    graph['input']   = tf.Variable(np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)), dtype = 'float32')
    graph['conv1_1']  = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    return graph

###############################################################################
#定义的内容损失函数
def content_loss_func(sess, model):
    def _content_loss(p, x):
        N = p.shape[3]#N是滤波器的个数
        M = p.shape[1] * p.shape[2]#高度乘以feature map的宽度.
        #复制与样式丢失中使用的相同的规范化常量
        return (1 / (4 * N * M)) * tf.reduce_sum(tf.pow(x - p, 2))
    return _content_loss(sess.run(model['conv4_2']), model['conv4_2'])

###############################################################################
#定义的样式丢失函数
def style_loss_func(sess, model):
    def _gram_matrix(F, N, M):
        Ft = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(Ft), Ft)
    def _style_loss(a, x):
        # 样式丢失计算
        N = a.shape[3]#意义同上
        M = a.shape[1] * a.shape[2]
        A = _gram_matrix(a, N, M)# 原始图像的样式表示
        G = _gram_matrix(x, N, M) # 生成图像的样式表示
        result = (1 / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow(G - A, 2))#在 tensorflow里求和函数
        return result
    
 ###############################################################################
 #层使用
    layers = [
        ('conv1_1', 0.5),
        ('conv2_1', 1.0),
        ('conv3_1', 1.5),
        ('conv4_1', 3.0),
        ('conv5_1', 4.0),
    ]
    E = [_style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in layers]
    W = [w for _, w in layers]
    loss = sum([W[l] * E[l] for l in range(len(layers))])
    return loss

###############################################################################
if __name__ == '__main__':
    with tf.Session() as sess:
        #加载图片
        content_image = load_image(CONTENT_IMAGE)
        style_image = load_image(STYLE_IMAGE)
        # 加载vgg模型
        model = load_vgg_model(VGG_MODEL)
        # 生成样式与内容表示混合图像，也就是开始画画
        input_image = generate_noise_image(content_image)
        sess.run(tf.initialize_all_variables())
        # 用content_image构建content_loss
        sess.run(model['input'].assign(content_image))
        content_loss = content_loss_func(sess, model)
        # 用style_image构建style_loss
        sess.run(model['input'].assign(style_image))
        style_loss = style_loss_func(sess, model)
###############################################################################
        # 实例
        total_loss = BETA * content_loss + ALPHA * style_loss
        # 内容由一层构建，而样式由五层构建
        optimizer = tf.train.AdamOptimizer(2.0)
        train_step = optimizer.minimize(total_loss)
        sess.run(tf.initialize_all_variables())
        sess.run(model['input'].assign(input_image))
        for it in range(ITERATIONS):
            sess.run(train_step)
            if it%100 == 0:
                # 每100次迭代打印一次.
                mixed_image = sess.run(model['input'])
                print('Iteration %d' % (it))
                print('sum : ', sess.run(tf.reduce_sum(mixed_image)))
                print('cost: ', sess.run(total_loss))

                if not os.path.exists(OUTPUT_DIR):#文件的属性获取
                    os.mkdir(OUTPUT_DIR)#创建路径
                filename = 'output/%d.png' % (it)
                save_image(filename, mixed_image)#保存