# YOLO11 + Google Coral on Raspberry Pi 4

Real‑time object detection using YOLO11 and the Google Coral USB Accelerator on a Raspberry Pi 4.

## Features
- Accelerated inference with Coral Edge TPU (15‑25 FPS)
- Supports USB camera, video files, or individual images
- Adjustable confidence threshold and display resolution
- Optional recording of output video

## Hardware Requirements
- Raspberry Pi 4 (2GB+ RAM)
- Google Coral USB Accelerator
- USB webcam (or Pi Camera)

## Software Requirements (see `requirements.txt`)
- Python 3.9
- Ultralytics YOLO11
- tflite‑runtime 2.9.1
- OpenCV
