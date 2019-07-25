
# coding: utf-8

# In[1]:


import tensorflow as tf
import sonnet as snt

from PIL import Image, ImageOps
import cv2

import numpy as np

import os

import i3d


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
    new_img = img_scale[(new_h-224)/2:(new_h+224)/2,(new_w-224)/2:(new_w+224)/2,:]
    
    return new_img

def reshape_cv2(img, type):
    width, height = img.shape[0:2]
    min_ = min(height, width)
    ratio = float(256/float(min_))
    new_w = int(ratio*width)
    new_h = int(ratio*height)
    if type=='rgb':
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (new_w,new_h), interpolation=cv2.INTER_LINEAR)
    frame = (frame/255.0)*2-1
    frame = frame[(new_h-224)/2:(new_h+224)/2,(new_w-224)/2:(new_w+224)/2]
    
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


# In[4]:


image_size = 224
num_class = 20

sample_path = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

checkpoints = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

def extract_feat():
    
    rgb_input = tf.placeholder(tf.float32, shape=(1,None,image_size,image_size,3))
    flow_input = tf.placeholder(tf.float32, shape=(1,None,image_size,image_size,2))
    with tf.variable_scope('RGB'):
        rgb_model = i3d.InceptionI3d(num_class+1, spatial_squeeze=True, final_endpoint='Mixed_5c')
        rgb_mixed5c, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)
        rgb_feat = tf.nn.avg_pool3d(rgb_mixed5c, ksize=[1, 2, 7, 7, 1],
                             strides=[1, 1, 1, 1, 1], padding=snt.VALID)

    rgb_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'RGB':
            rgb_variable_map[variable.name.replace(':0', '')] = variable
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
    
    with tf.variable_scope('Flow'):
        flow_model = i3d.InceptionI3d(num_class+1,spatial_squeeze=True, final_endpoint='Mixed_5c')
        flow_mixed5c, _ = flow_model(flow_input, is_training=False, dropout_keep_prob=1.0)
        flow_feat = tf.nn.avg_pool3d(flow_mixed5c, ksize=[1, 2, 7, 7, 1],
                             strides=[1, 1, 1, 1, 1], padding=snt.VALID)
        
    flow_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'Flow':
            flow_variable_map[variable.name.replace(':0', '')] = variable
    flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
    
    with tf.Session() as sess:
        feed_dict = {}
        
        rgb_saver.restore(sess, checkpoints['rgb_imagenet'])
        flow_saver.restore(sess, checkpoints['flow_imagenet'])
        
        tf.logging.info('RGB checkpoint restored')
        tf.logging.info('Flow checkpoint restored')
       
        val_path = '/home/june/mnt2/th14_feature_i3d/th14_raw/val_optical_flow_rgb'
#         test_path = '/home/june/mnt2/th14_feature_i3d/th14_raw/test_optical_flow_rgb'
        
        save_path = '/home/june/mnt2/th14_feature_i3d/th14_feature/val_feat'

        feat_step = 16

        video_list = os.listdir(val_path)
#         print(len(video_list))
        for video in video_list:
            print(video)
            video_path = os.path.join(val_path, video)

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
                rgb = np.squeeze(rgb)
                flow = np.squeeze(flow)
                save_name = video+'.mp4_'+str(float(idx+1))+'_'+str(float(str(idx+1+feat_step)))+'.npy'
                print(save_name)
                np.save(os.path.join(save_path, 'rgb', save_name), rgb)
                np.save(os.path.join(save_path, 'flow', save_name), flow)
#                 print(rgb.shape, flow.shape)
                
            
        
#         return rgb, flow


# In[ ]:


extract_feat()

