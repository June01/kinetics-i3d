#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import sonnet as snt

from PIL import Image, ImageOps
import cv2

import numpy as np

import os

import i3d

import sys

inp1 = sys.argv[1]
inp2 = sys.argv[2]

# In[2]:


# Proprecessing for image(scale and crop)
def reshape_img_pil(img):
    width, height = np.array(img).shape[0:2]
    min_ = min(height, width)
    ratio = float(256/float(min_))
    new_w = int(ratio*width)
    new_h = int(ratio*height)
    
    img_resize = np.array(img.resize((new_w, new_h), resample=Image.BILINEAR))
    img_scale = (img_resize/255.0)*2-1
    new_img = img_scale[int((new_h-224)/2):int((new_h+224)/2),int((new_w-224)/2):int((new_w+224)/2),:]
    
    return new_img

def reshape_cv2(img, type):
    width, height = img.shape[0:2]
    min_ = min(height, width)
    ratio = float(256/float(min_))
    new_w = int(ratio*width)
    new_h = int(ratio*height)
#     print(width, height, new_w, new_h)
#     print((new_h-224)/2, (new_h+224)/2, (new_w-224)/2, (new_w+224)/2)
    if type=='rgb':
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (new_w,new_h), interpolation=cv2.INTER_LINEAR)
    frame = (frame/255.0)*2-1
    frame = frame[int((new_h-224)/2):int((new_h+224)/2),int((new_w-224)/2):int((new_w+224)/2)]
    
    return frame


# In[3]:


def get_batch(idx, step, video_path, video_name, type):
    raw_images = []
    for i in range(step):
        if type == 'rgb':
            image_name = 'img_%05d.jpg'%(idx+1+i)
            if os.path.exists(os.path.join(video_path, image_name)):
                img = cv2.imread(os.path.join(video_path, image_name))
                img = reshape_cv2(img, type='rgb')
                raw_images.append(img)
        elif type == 'flow':
            flow_x_name = 'flow_x_%05d.jpg'%(idx+1+i)
            flow_y_name = 'flow_y_%05d.jpg'%(idx+1+i)
            if os.path.exists(os.path.join(video_path, flow_x_name)):
                flow_x_img = cv2.imread(os.path.join(video_path, flow_x_name))
                flow_y_img = cv2.imread(os.path.join(video_path, flow_y_name))
                
                flow_x_img = reshape_cv2(flow_x_img, type='flow')
                flow_y_img = reshape_cv2(flow_y_img, type='flow')
                
#                 print(flow_x_img.shape, flow_y_img.shape)
#                 flow = np.stack((flow_x_img, flow_y_img))
#                 print(flow.shape)
                flow = np.stack((flow_x_img, flow_y_img)).reshape(224,224,2)

                raw_images.append(flow)
    
    return np.array(raw_images)


# In[13]:


image_size = 224
num_class = 20

sample_path = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

checkpoints = {
    'rgb_scratch': 'data/checkpoints/rgb_scratch/model.ckpt',
    'flow_scratch': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

raw_path = {
    'val': '/data/th14_raw/val_optical_flow_rgb',
    'test': '/data/th14_raw/test_optical_flow_rgb',
}

save_paths = {
    'val_imagenet': '/data/th14_feature_i3d/feat_and_var/feat_imagenet/val_feat',
    'test_imagenet': '/data/th14_feature_i3d/feat_and_var/feat_imagenet/test_feat',
    'val_scratch': '/data/th14_feature_i3d/feat_and_var/feat_scratch/val_feat',
    'test_scratch': '/data/th14_feature_i3d/feat_and_var/feat_scratch/test_feat',
}


# In[4]:


rgb_input = tf.placeholder(tf.float32, shape=(1,None,image_size,image_size,3))
flow_input = tf.placeholder(tf.float32, shape=(1,None,image_size,image_size,2))
with tf.variable_scope('RGB'):
    rgb_model = i3d.InceptionI3d(num_class+1, spatial_squeeze=True, final_endpoint='Mixed_5c')
    rgb_mixed5c, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)
#         rgb_feat = tf.nn.avg_pool3d(rgb_mixed5c, ksize=[1, 2, 7, 7, 1],
#                              strides=[1, 1, 1, 1, 1], padding=snt.VALID)
    rgb_feat = rgb_mixed5c

rgb_variable_map = {}
for variable in tf.global_variables():
    if variable.name.split('/')[0] == 'RGB':
        rgb_variable_map[variable.name.replace(':0', '')] = variable
rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
    
with tf.variable_scope('Flow'):
    flow_model = i3d.InceptionI3d(num_class+1,spatial_squeeze=True, final_endpoint='Mixed_5c')
    flow_mixed5c, _ = flow_model(flow_input, is_training=False, dropout_keep_prob=1.0)
#         flow_feat = tf.nn.avg_pool3d(flow_mixed5c, ksize=[1, 2, 7, 7, 1],
#                              strides=[1, 1, 1, 1, 1], padding=snt.VALID)
    flow_feat = flow_mixed5c
        
flow_variable_map = {}
for variable in tf.global_variables():
    if variable.name.split('/')[0] == 'Flow':
        flow_variable_map[variable.name.replace(':0', '')] = variable
flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)


# In[9]:


def get_mean_var(feat):
    feat = np.reshape(feat, (-1, 1024))
    mean = np.mean(feat, axis=0)
    var = np.var(feat, axis=0)
    feat_all = np.hstack((mean, var))
    return feat_all


# In[18]:


def extract_feat(feat_extractor='imagenet', data_source='test'):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2) # 30% memory of TITAN is enough
#    self.sess = tf.Session(config = tf.ConfigProto(gpu_options=gpu_options))    
    with tf.Session(config = tf.ConfigProto(gpu_options=gpu_options)) as sess:
        feed_dict = {}
        
        rgb_feat_type = 'rgb' + '_' + feat_extractor
        flow_feat_type = 'flow' + '_' + feat_extractor
        
        rgb_saver.restore(sess, checkpoints[rgb_feat_type])
        flow_saver.restore(sess, checkpoints[flow_feat_type])
#         rgb_saver.restore(sess, checkpoints['rgb'])
#         flow_saver.restore(sess, checkpoints['flow'])
        
        tf.logging.info('RGB checkpoint restored')
        tf.logging.info('Flow checkpoint restored')
       
        feat_path = raw_path[data_source]
        
        save_pn = data_source + '_' + feat_extractor
        save_path = save_paths[save_pn]

        feat_step = 16

        video_list = os.listdir(feat_path)
#         print(len(video_list))
        for video in video_list:
#             video = 'video_test_0001292'
            
            video_path = os.path.join(feat_path, video)
#             if not os.path.exists(video_path):
#                 os.makedirs(video_path)
            print(video_path)
            num_frames = len(os.listdir(video_path))/3
            index = np.arange(num_frames-8, step=8)
#             print(len(index))
            for idx in index:
                rgb_batch = get_batch(idx, feat_step, video_path, video, type='rgb')
                flow_batch = get_batch(idx, feat_step, video_path, video, type='flow')

                rgb_arr = rgb_batch[np.newaxis, :]
#                 rgb_arr = (rgb_arr/255.0)*2-1
                flow_arr = flow_batch[np.newaxis, :]
#                 flow_arr = (flow_arr/255.0)*2-1

                feed_dict[rgb_input] = rgb_arr
                feed_dict[flow_input] = flow_arr

                rgb, flow = sess.run([rgb_feat, flow_feat], feed_dict=feed_dict)
#                 print(rgb.shape, flow.shape)
                rgb = get_mean_var(rgb)
                flow = get_mean_var(flow)
                print(rgb.shape, flow.shape)
                save_name = video+'.mp4_'+str(float(idx+1))+'_'+str(float(str(idx+1+feat_step)))+'.npy'
                print(save_path,save_name)
                np.save(os.path.join(save_path, 'rgb', save_name), rgb)
                np.save(os.path.join(save_path, 'flow', save_name), flow)
                
#             break
                


# In[19]:


extract_feat(feat_extractor=inp1, data_source=inp2)


# In[ ]:





# In[ ]:




