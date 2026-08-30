import tensorflow as tf
from keras import layers
#Augmentation layer to change slight fundamentals of the image 
#Rotate, Flip ,etc. to better train the model
def get_augmentation_layer():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"), #Flip image horizontally
        layers.RandomRotation(0.1),      #Rotate image by 36 degree
        layers.RandomZoom(0.1),          #Zoom image by 10%
        layers.RandomContrast(0.1),      #Contrast image by 10%
    ], name="augmentation")
 