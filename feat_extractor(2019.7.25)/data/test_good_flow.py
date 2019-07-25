import numpy as np
import tensorflow as tf

import cv2

gt = np.load('v_CricketShot_g04_c01_flow.npy') 

cap = cv2.VideoCapture('v_CricketShot_g04_c01.avi')
frames = []

if cap.isOpened() == False:
    print('Error opening video stream or file')

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (224,224), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    else:
        break
cap.release()
frames = np.array(frames)

optical_flow = cv2.DualTVL1OpticalFlow_create()

flow_set = []
for i in range(len(frames)-1):
    prev = frames[i]
    now = frames[i+1]
    flow = optical_flow.calc(prev, now, None)
    flow_set.append(flow)

flows = np.array(flow_set)
flows = flows/20.0
print(flows.shape)
print(gt.shape)
flows_flatten = flows.flatten()
gt_flatten = gt.flatten()

print(np.linalg.norm(gt_flatten-flows_flatten))

