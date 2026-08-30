import tensorflow as tf
from keras import Model, layers
from keras import models
from keras.applications import EfficientNetB0
from keras.applications.efficientnet import preprocess_input

from preprocessing.augmentation import get_augmentation_layer
#Building the model
def build_model(input_shape=(224,224,3)):
    base_model= EfficientNetB0(
        include_top=False,  #False to remove pre-defined classification of EfficientNetB0 and train custom weights
        weights="imagenet", 
        input_shape=input_shape
    )
    base_model.trainable=False  #Freeze the base model to prevent its weights from being updated during training
    inputs=tf.keras.Input(shape=input_shape)
    x = get_augmentation_layer()(inputs)      # random augmentation, train-only
    x = preprocess_input(x)                   # EfficientNet-specific normalization
    x = base_model(x, training=False)         # run base in inference mode extracting image features
    x = layers.GlobalAveragePooling2D()(x)    # Convert feature maps into a feature vector
    x = layers.Dropout(0.2)(x)                #Reduce Overfitting by disabling 20% of neurons while training
    outputs = layers.Dense(1, activation="sigmoid")(x) #Binary classification via Sigmoid function

    model = Model(inputs, outputs)
    return model, base_model
def compile_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",  #Loss for Binary Classification
        metrics=["accuracy"]         #Track Accuracy
    )
    return model

#Generate model summary when run as a script
if __name__ == "__main__":
    model, base_model = build_model()
    model = compile_model(model)
    model.summary()