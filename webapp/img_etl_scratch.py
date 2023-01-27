import cv2
import tensorflow as tf

## Load in images:
img1 = cv2.imread("CXR2_IM-0652-2001.png", cv2.IMREAD_UNCHANGED)
img2 = cv2.imread("CXR3_IM-1384-1001.png", cv2.IMREAD_UNCHANGED)

## Load TFlite Model
TFLITE_FILE_PATH = 'Chest_Xray_caption_generator.tflite'
interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)

