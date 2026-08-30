import os
import tensorflow as tf
#Loading images and creating datasets for training, validation, and testing
#(224,224) -> Standard size of image for EfficientNetB0
def get_datasets(data_dir, image_size=(224, 224), batch_size=64):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir + "/train",  #data_dir="/path/to/train"
        image_size=image_size,
        batch_size=batch_size,
        label_mode='binary', #Assigns labels based on the directory structure (0 for fake, 1 for real)
        shuffle=True,
        seed=42 
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir + "/valid",  #data_dir="/path/to/valid"
        image_size=image_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir + "/test",   #data_dir="/path/to/test" 
        image_size=image_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=False
    )
    #checking the class names and labels 0=fake, 1=real
    class_names = train_ds.class_names
    print(f"Class names (label 0, label 1): {class_names}")
    #Prefetching the datasets for performance optimization
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE) #AUTOTUNE allows dynamic allocation for better performance
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)     #AUTOTUNE reloads the next batch while the current batch 
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)   #Occurs when first batch is being processed
    return train_ds, val_ds, test_ds
#Work only when run directly, not when imported as a module
if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_datasets(
        data_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "processed") #Can be changed to DeepFake/dataset/<name> to use images there
    )
#Optional
    # Checking first batch
    for images, labels in train_ds.take(1):
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)