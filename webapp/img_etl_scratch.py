import cv2
import pathlib
import numpy as np
import tensorflow as tf

## Load in images:
img1 = cv2.imread("CXR2_IM-0652-2001.png", cv2.IMREAD_UNCHANGED)/255
img1.dtype = 'float32'
img1 = cv2.resize(img1, (224, 224), interpolation = cv2.INTER_NEAREST)
img2 = cv2.imread("CXR3_IM-1384-1001.png", cv2.IMREAD_UNCHANGED)/255
img2.dtype = 'float32'
## Load TFlite Model
TFLITE_FILE_PATH = 'Chest_Xray_caption_generator.tflite'
interpreter = tf.lite.Interpreter(model_path = TFLITE_FILE_PATH)

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()

# # input details
# print(input_details)
# # output details
# print(output_details)

interpreter.set_tensor(input_details[0]["index"], [img1])

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])

