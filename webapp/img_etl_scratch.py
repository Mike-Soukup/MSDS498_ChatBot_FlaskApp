import cv2
import pathlib
import numpy as np
import tensorflow as tf

## Define prediction function:
def greedy_search_predict(image1,image2,model):
  """
  Given paths to two x-ray images predicts the impression part of the x-ray in a greedy search algorithm
  """
  image1 = cv2.imread(image1,cv2.IMREAD_UNCHANGED)/255 
  image2 = cv2.imread(image2,cv2.IMREAD_UNCHANGED)/255
  image1 = tf.expand_dims(cv2.resize(image1,input_size,interpolation = cv2.INTER_NEAREST),axis=0) #introduce batch and resize
  image2 = tf.expand_dims(cv2.resize(image2,input_size,interpolation = cv2.INTER_NEAREST),axis=0)
  image1 = model.get_layer('image_encoder')(image1)
  image2 = model.get_layer('image_encoder')(image2)
  image1 = model.get_layer('bkdense')(image1)
  image2 = model.get_layer('bkdense')(image2)

  concat = model.get_layer('concatenate')([image1,image2])
  enc_op = model.get_layer('encoder_batch_norm')(concat)  
  enc_op = model.get_layer('encoder_dropout')(enc_op) #this is the output from encoder


  decoder_h,decoder_c = tf.zeros_like(enc_op[:,0]),tf.zeros_like(enc_op[:,0])
  a = []
  pred = []
  for i in range(max_pad):
    if i==0: #if first word
      caption = np.array(tokenizer.texts_to_sequences(['<cls>'])) #shape: (1,1)
    output,decoder_h,attention_weights = model.get_layer('decoder').onestepdecoder(caption,enc_op,decoder_h)#,decoder_c) decoder_c,

    #prediction
    max_prob = tf.argmax(output,axis=-1)  #tf.Tensor of shape = (1,1)
    caption = np.array([max_prob]) #will be sent to onstepdecoder for next iteration
    if max_prob==np.squeeze(tokenizer.texts_to_sequences(['<end>'])): 
      break;
    else:
      a.append(tf.squeeze(max_prob).numpy())
  return tokenizer.sequences_to_texts([a])[0] #here output would be 1,1 so subscripting to open the array

## Load in images:
img1 = cv2.imread("CXR2_IM-0652-2001.png", cv2.IMREAD_UNCHANGED)/255
img1.dtype = 'float32'
img1 = cv2.resize(img1, (224, 224), interpolation = cv2.INTER_NEAREST)
img2 = cv2.imread("CXR3_IM-1384-1001.png", cv2.IMREAD_UNCHANGED)/255
img2.dtype = 'float32'

## Load TFlite Model
TFLITE_FILE_PATH = 'Chest_Xray_caption_generator.tflite'
model = tf.lite.Interpreter(model_path = TFLITE_FILE_PATH)

# Get input and output tensors.
input_details = model.get_input_details()
output_details = model.get_output_details()

model.allocate_tensors()


greedy_search_predict(img1,img2,model = model)

# # input details
# print(input_details)
# # output details
# print(output_details)

# interpreter.set_tensor(input_details[0]["index"], [img1])

# interpreter.invoke()

# output_data = interpreter.get_tensor(output_details[0]['index'])

