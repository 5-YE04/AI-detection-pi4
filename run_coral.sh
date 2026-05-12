#!/bin/bash
source ~/yolo_coral_env/bin/activate
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
python detect_coral.py --model /home/batman/yolo11n_full_integer_quant_edgetpu.tflite --source usb0 --thresh 0.5 --resolution 640x480