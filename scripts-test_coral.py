#!/usr/bin/env python3
import tflite_runtime.interpreter as tflite
import numpy as np

delegate = tflite.load_delegate('libedgetpu.so.1')
interpreter = tflite.Interpreter(
    model_path='/home/batman/yolo11n_full_integer_quant_edgetpu.tflite',
    experimental_delegates=[delegate]
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
print('Model input shape:', input_details[0]['shape'])
print('Coral is ready!')