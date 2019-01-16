from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time
import logging
import numpy as np
import tensorflow as tf

import i3d
from rmb_lib.action_dataset import *
from rmb_lib.label_trans import *


_FRAME_SIZE = 224 
_CLIP_SIZE = 64

_CHECKPOINT_PATHS = {
    'rgb': './model/rgb/ucf101_rgb_0.946_model-44520',
    'flow': './model/flow/ucf101_flow_0.963_model-28620'
}

_CHANNEL = {
    'rgb': 3,
    'flow': 2,
}

_SCOPE = {
    'rgb': 'RGB',
    'flow': 'Flow',
}

_CLASS_NUM = {
    'ucf101': 101,
    'hmdb51': 51
}

log_dir = './error_record'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

def load_model(sess, saver, checkpoint_dir):
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and ckpt.model_checkpoint_path:
    print("train model : " + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

def main(dataset_name, data_tag, stream):
    assert data_tag in ['rgb', 'flow', 'mixed']

    logging.basicConfig(level=logging.INFO, filename=os.path.join(log_dir, 'log_{}.txt'.format(stream)), filemode='w', format='%(message)s')

    model_save_dir = './model/{}'.format(stream)

    if data_tag == 'flow': checkpoint = 'flow_imagenet'
    else: checkpoint = 'rgb600'

    kinetics_classes = 400
    if checkpoint == 'rgb600': kinetics_classes = 600
    
    label_map = get_label_map(os.path.join('data', dataset_name, 'label_map.txt'))

    if data_tag in ['rgb', 'mixed']:
        _, test_info = split_data(
            os.path.join('./data', dataset_name, '{}.txt'.format(stream)),
            os.path.join('./data', dataset_name, 'testlist01.txt'))
    if data_tag in ['flow', 'mixed']:
        _, test_info = split_data(
            os.path.join('./data', dataset_name, '{}.txt'.format(stream)),
            os.path.join('./data', dataset_name, 'testlist01.txt'))


    label_holder = tf.placeholder(tf.int32, [None])
   
    if data_tag in ['rgb', 'mixed']:
        rgb_data = Action_Dataset(dataset_name, 'rgb', stream, test_info)
        rgb_holder = tf.placeholder(
            tf.float32, [None, None, _FRAME_SIZE, _FRAME_SIZE, _CHANNEL['rgb']])
    if data_tag in ['flow', 'mixed']:
        flow_data = Action_Dataset(dataset_name, 'flow', stream, test_info)
        flow_holder = tf.placeholder(
            tf.float32, [None, None, _FRAME_SIZE, _FRAME_SIZE, _CHANNEL['flow']])


    if data_tag in ['rgb', 'mixed']:
        with tf.variable_scope(_SCOPE['rgb']):
            rgb_model = i3d.InceptionI3d(kinetics_classes, spatial_squeeze=True, final_endpoint='Logits')
            rgb_logits, _ = rgb_model(rgb_holder, is_training=False, dropout_keep_prob=1)
            rgb_logits_dropout = tf.nn.dropout(rgb_logits, 1)   
            rgb_fc_out = tf.layers.dense(rgb_logits_dropout, _CLASS_NUM[dataset_name], use_bias=True)
            softmax = tf.nn.softmax(rgb_fc_out)
            rgb_top_k_op = tf.nn.in_top_k(rgb_fc_out, label_holder, 1)

    if data_tag in ['flow', 'mixed']:
        with tf.variable_scope(_SCOPE['flow']):
            flow_model = i3d.InceptionI3d(kinetics_classes, spatial_squeeze=True, final_endpoint='Logits')
            flow_logits, _ = flow_model(flow_holder, is_training=False, dropout_keep_prob=1)
            flow_logits_dropout = tf.nn.dropout(flow_logits, 1)   
            flow_fc_out = tf.layers.dense(flow_logits_dropout, _CLASS_NUM[dataset_name], use_bias=True)
            softmax = tf.nn.softmax(flow_fc_out)
            flow_top_k_op = tf.nn.in_top_k(flow_fc_out, label_holder, 1)

    variable_map = {}
    if data_tag in ['rgb', 'mixed']:
        for variable in tf.global_variables():
            tmp = variable.name.split('/')
            if tmp[0] == _SCOPE['rgb']:   
                if checkpoint == 'rgb600':
                    variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
                else:
                    variable_map[variable.name.replace(':0', '')] = variable
        #rgb_saver = tf.train.Saver(var_list=variable_map)
        rgb_saver = tf.train.Saver(max_to_keep=10)

    variable_map = {}
    if data_tag in ['flow', 'mixed']:
        for variable in tf.global_variables():
            tmp = variable.name.split('/')
            if tmp[0] == _SCOPE['flow']:
                variable_map[variable.name.replace(':0', '')]=variable
        flow_saver = tf.train.Saver(var_list=variable_map, reshape=True)

    
    if data_tag == 'rgb':
        fc_out = rgb_fc_out
    if data_tag == 'flow':
        fc_out = flow_fc_out
    if data_tag == 'mixed':
        fc_out = rgb_fc_out + flow_fc_out

    softmax = tf.nn.softmax(fc_out)
    top_k_op = tf.nn.in_top_k(fc_out, label_holder, 1)
    
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    # sess = tf.Session(config=config)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    print("==============================================")
    if data_tag in ['rgb', 'mixed']:
        #rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
        load_model(sess, rgb_saver, model_save_dir)    

    if data_tag in ['flow', 'mixed']:
        #flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
        load_model(sess, flow_saver, model_save_dir)    

    print("Start {} stream test!".format(stream))
    print("==============================================")
    true_count = 0
    video_size = len(test_info)
    error_record = open(os.path.join(log_dir, 'error_record_{}.txt'.format(stream)), 'w')
    f = open("./result/predict_ret_{}.txt".format(stream), 'w')
    f_fc = open("./result/fc_{}.txt".format(stream), 'w')

    for i in range(video_size):
        feed_dict = {}
        if data_tag in ['rgb', 'mixed']:
            rgb_clip, label = rgb_data.next_batch(
                1, rgb_data.videos[i].total_frame_num, shuffle=False, data_augment=False)
            rgb_clip = rgb_clip/255
            clip = rgb_clip
            feed_dict[rgb_holder] = rgb_clip
            video_name = rgb_data.videos[i].name
        if data_tag in ['flow', 'mixed']:
            flow_clip, label = flow_data.next_batch(
                1, flow_data.videos[i].total_frame_num, shuffle=False, data_augment=False)
            flow_clip = 2*(flow_clip/255)-1
            clip = flow_clip
            feed_dict[flow_holder] = flow_clip
            video_name = flow_data.videos[i].name
        feed_dict[label_holder] = label
        top_1, predictions, fc = sess.run([top_k_op, softmax, fc_out], feed_dict=feed_dict)
        tmp = np.sum(top_1)
        true_count += tmp
        print('Video%d: %d, accuracy: %.4f (%d/%d), name:%s' % (i+1, tmp, true_count/(i+1), true_count, i+1, video_name))
        logging.info('Video%d: %d, accuracy: %.4f (%d/%d) , name:%s' % (i+1, tmp, true_count/(i+1), true_count, i+1, video_name))
        if tmp==0:
            wrong_answer = np.argmax(predictions, axis=1)[0]
            print('---->answer: {}, wrong answer: {}, probability: {:.2f}'.format(label[0], wrong_answer,  predictions[0, wrong_answer]))
            logging.info('---->answer: {}, probability: {:.2f}'.format(label[0], wrong_answer,  predictions[0, wrong_answer]))
            error_record.write(
                'video: {}, answer: {}, wrong answer: {}, probability: {:.2f}\n'.format(video_name, label[0], wrong_answer, predictions[0, wrong_answer]))
        
        #print(predictions[0])
        #print(top_1)
        #print(label[0], np.argmax(predictions, axis=1)[0])
        f.write('{}\t{}'.format(label[0], np.argmax(predictions, axis=1)[0]))
        for p in predictions[0]:
            f.write('\t{}'.format(p))
        f.write('\n')

        for fc_ in fc[0]:
            f_fc.write('{}\t'.format(fc_))
        f_fc.write('\n')

    error_record.close()
    f.close()
    f_fc.close()
    accuracy = true_count/ video_size
    print('test accuracy: %.4f' % (accuracy))
    logging.info('test accuracy: %.4f' % (accuracy))

    print("done")
    sess.close()
   

if __name__ == '__main__':
    description = 'finetune on other dataset'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('dataset_name', type=str)
    p.add_argument('data_tag', type=str)
    p.add_argument('stream', type=str)
    main(**vars(p.parse_args()))
