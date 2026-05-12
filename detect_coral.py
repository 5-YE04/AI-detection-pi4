#!/usr/bin/env python3
"""
YOLO11 + Google Coral Edge TPU detection script.
Uses --device='tpu' for accelerated inference.
"""

import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True,
                    help='Path to .tflite Edge TPU model')
parser.add_argument('--source', required=True,
                    help='Image source: image file, folder, video, usb0, picamera0')
parser.add_argument('--thresh', default=0.5, type=float,
                    help='Confidence threshold')
parser.add_argument('--resolution', default=None,
                    help='Display resolution (WxH)')
parser.add_argument('--record', action='store_true',
                    help='Record output to demo1.avi')

args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

if not os.path.exists(model_path):
    print('ERROR: Model file not found.')
    sys.exit(1)

# Load model – device='tpu' is the key for Coral acceleration
model = YOLO(model_path, task='detect')
labels = model.names

img_ext_list = ['.jpg','.jpeg','.png','.bmp']
vid_ext_list = ['.avi','.mov','.mp4','.mkv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    ext = os.path.splitext(img_source)[1].lower()
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print('Unsupported file type')
        sys.exit(1)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print('Invalid source')
    sys.exit(1)

resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video/camera')
        sys.exit(1)
    if not user_res:
        print('Must specify --resolution for recording')
        sys.exit(1)
    record_fps = 30
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'),
                               record_fps, (resW, resH))

if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = glob.glob(img_source + '/*')
    imgs_list = [f for f in imgs_list if os.path.splitext(f)[1].lower() in img_ext_list]
elif source_type in ('video','usb'):
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

avg_fps = 0
fps_buffer = []
img_count = 0

while True:
    t_start = time.perf_counter()

    # Load frame
    if source_type in ('image','folder'):
        if img_count >= len(imgs_list):
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            break
    elif source_type == 'usb':
        ret, frame = cap.read()
        if frame is None or not ret:
            print('Camera read error')
            break
    elif source_type == 'picamera':
        frame = cap.capture_array()
        if frame is None:
            break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Inference with Coral TPU
    results = model(frame, verbose=False, device='tpu')

    detections = results[0].boxes
    object_count = 0

    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()
        if conf > min_thresh:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_label = max(ymin, label_h + 10)
            cv2.rectangle(frame, (xmin, y_label-label_h-10), (xmin+label_w, y_label+baseline-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, y_label-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            object_count += 1

    if source_type in ('video','usb','picamera'):
        cv2.putText(frame, f'FPS: {avg_fps:.1f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f'Objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.imshow('YOLO Coral Detection', frame)
    if record:
        recorder.write(frame)

    key = cv2.waitKey(1 if source_type in ('video','usb','picamera') else 0)
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.waitKey()
    elif key == ord('p'):
        cv2.imwrite('capture.png', frame)

    t_stop = time.perf_counter()
    fps = 1 / (t_stop - t_start)
    fps_buffer.append(fps)
    if len(fps_buffer) > 200:
        fps_buffer.pop(0)
    avg_fps = np.mean(fps_buffer)

print(f'Average FPS: {avg_fps:.2f}')
if source_type in ('video','usb'):
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()