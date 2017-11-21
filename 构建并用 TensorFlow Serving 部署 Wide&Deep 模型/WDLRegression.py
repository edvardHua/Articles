#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18/09/2017 4:02 PM
# @Author  : zengzihua
# @Software: PyCharm

from os.path import join

from pprint import pprint
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

import tensorflow as tf
import numpy as np
import math
import random


def deep_model(input_data, hidden1_units, hidden2_units, hidden3_units):
    """
    三层的神经网络
    :param input_data: 2-D tensor
    :param hidden1_units: int
    :param hidden2_units: int
    :param hidden3_units: int
    :return: 
    """
    # 得到每个样本的维度
    input_len = int(input_data.shape[1])
    with tf.name_scope("hidden1"):
        # 修正的方式初始化权重
        weights = tf.Variable(tf.truncated_normal([input_len, hidden1_units],
                                                  stddev=1.0 / math.sqrt(float(input_len))
                                                  ), name="weights1"
                              )
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases1')
        hidden1 = tf.nn.relu(tf.matmul(input_data, weights)) + biases
    with tf.name_scope("hidden2"):
        # 修正的方式初始化权重
        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units],
                                                  stddev=1.0 / math.sqrt(float(input_len))
                                                  ), name="weights2"
                              )
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases2')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    with tf.name_scope("hidden3"):
        # 修正的方式初始化权重
        weights = tf.Variable(tf.truncated_normal([hidden2_units, hidden3_units],
                                                  stddev=1.0 / math.sqrt(float(input_len))
                                                  ), name="weights3"
                              )
        biases = tf.Variable(tf.zeros([hidden3_units]), name='biases3')
        hidden3 = tf.nn.relu(tf.matmul(hidden2, weights) + biases)

    with tf.name_scope("output"):
        # 修正的方式初始化权重
        weights = tf.Variable(tf.truncated_normal([hidden3_units, 1],
                                                  stddev=1.0 / math.sqrt(float(input_len))
                                                  ), name="weights4"
                              )
        biases = tf.Variable(tf.zeros([1]), name='biases4')
        output = tf.nn.relu(tf.matmul(hidden3, weights) + biases)

    return tf.nn.relu(output)


def wide_model(input_data):
    """
    一层的神经网络，相当于是 LR
    :param input_data: 
    :return: 
    """
    input_len = int(input_data.shape[1])
    with tf.name_scope("wide"):
        # 修正的方式初始化权重，输出层结点只有一个
        weights = tf.Variable(tf.truncated_normal([input_len, 1],
                                                  stddev=1.0 / math.sqrt(float(input_len))
                                                  ), name="weights"
                              )
        output = tf.matmul(input_data, weights)
        # 沿着行这个纬度来求和
        output = tf.reduce_sum(output, 1, name="reduce_sum")
        # 输出每个样本经过计算的值
        output = tf.reshape(output, [-1, 1])
    return output


def build_wdl(deep_input, wide_input, y):
    """
    得到模型和损失函数
    :param deep_input: 
    :param wide_input: 
    :param y: 
    :return: 
    """
    central_bias = tf.Variable([np.random.randn()], name="central_bias")
    dmodel = deep_model(deep_input, 256, 128, 64)
    wmodel = wide_model(wide_input)

    # 使用 LR 将两个模型组合在一起
    dmodel_weight = tf.Variable(tf.truncated_normal([1, 1]), name="dmodel_weight")
    wmodel_weight = tf.Variable(tf.truncated_normal([1, 1]), name="wmodel_weight")

    network = tf.add(
        tf.matmul(dmodel, dmodel_weight),
        tf.matmul(wmodel, wmodel_weight)
    )

    prediction = tf.add(network, central_bias)

    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=prediction)
    )
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    return train_step, loss, prediction


MODEL_VERSION = '1'
INPUT_WIDE_KEY = 'x_wide'
INPUT_DEEP_KEY = 'x_deep'
OUTPUT_KEY = 'output'
WIDE_DIM = 10
DEEP_DIM = 10


def build_and_saved_wdl():
    """
    训练并保存模型
    :return: 
    """
    # 训练数据
    x_deep_data = np.random.rand(10000)
    x_deep_data = x_deep_data.reshape(-1, 10)

    x_wide_data = np.random.rand(10000)
    x_wide_data = x_wide_data.reshape(-1, 10)

    x_deep = tf.placeholder(tf.float32, [None, 10])
    x_wide = tf.placeholder(tf.float32, [None, 10])
    y = tf.placeholder(tf.float32, [None, 1])

    y_data = np.array(
        [random.randint(0, 1) for i in range(1000)]
    )
    y_data = y_data.reshape(-1, 1)

    # 为了简单起见，这里没有验证集，也就没有验证集的 loss
    train_step, loss, prediction = build_wdl(x_deep, x_wide, y)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(train_step, feed_dict={x_deep: x_deep_data, x_wide: x_wide_data, y: y_data})

    # 将训练好的模型保存在当前的文件夹下
    builder = tf.saved_model.builder.SavedModelBuilder(join("./model_name", MODEL_VERSION))
    inputs = {
        "x_wide": tf.saved_model.utils.build_tensor_info(x_wide),
        "x_deep": tf.saved_model.utils.build_tensor_info(x_deep)
    }
    output = {"output": tf.saved_model.utils.build_tensor_info(prediction)}
    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=output,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature}
    )
    builder.save()
    pprint("Model Saved Succeed.")


def test_servable_api():
    """
    测试 API
    :return: 
    """
    # 随机产生 10 条测试数据
    x_deep_data = np.random.rand(100).reshape(-1, 10)

    x_wide_data = np.random.rand(100).reshape(-1, 10)

    channel = implementations.insecure_channel('127.0.0.1', int(5000))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # 发送请求
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'inception'
    request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    request.inputs[INPUT_WIDE_KEY].CopyFrom(
        tf.contrib.util.make_tensor_proto(x_wide_data, shape=[10, WIDE_DIM], dtype=tf.float32))
    request.inputs[INPUT_DEEP_KEY].CopyFrom(
        tf.contrib.util.make_tensor_proto(x_deep_data, shape=[10, DEEP_DIM], dtype=tf.float32))
    # 10 秒超时
    res = stub.Predict(request, 10.0)

    pprint(res.outputs[OUTPUT_KEY])


if __name__ == '__main__':
    # build_and_saved_wdl()
    test_servable_api()
