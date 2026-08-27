import numpy as np
from PIL import Image
import tensorflow as tf
from utils import config
#Loading the Model once at import time to avoid reloading it for every test
model = tf.keras.models.load_model(config.ModelPath_Best)

def preprocess_image(image:Image.Image,target_size=config.Image_Size):
#Accept Image, Resize it to the target size (224,224) 
    image=image.convert('RGB') #Convert to RGB (3) --> (224,224,3)
    image = image.resize(target_size)
    image_array = np.array(image) #Convert to Array
    image_array = np.expand_dims(image_array, axis=0) #Add Batch Dimension --> (1,224,224,3)
    return image_array

#Prediction for Image
def predict(image: Image.Image) -> dict:
#Accept the converted Image and determine if it is real or fake using the model
    processed = preprocess_image(image)
    raw_score = model.predict(processed, verbose=0)[0][0]
    if raw_score >= 0.5:
        label = "real"
        confidence = raw_score * 100
    else:
        label = "fake"
        confidence = (1 - raw_score) * 100
    return {
        "label": label,
        "confidence": round(float(confidence), 2),
        "raw_score": round(float(raw_score), 4)
    }

#ONLY TESTING CODE AFTER THIS

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    # Hide the empty root tkinter window that would otherwise pop up
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select an image to test",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        img = Image.open(file_path)
        result = predict(img)
        print(result)
    else:
        print("No file selected.")