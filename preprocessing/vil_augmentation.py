import os
import tensorflow as tf
import matplotlib.pyplot as plt
from data_load import get_datasets
from augmentation import get_augmentation_layer

# Optional function to preview image augmentation
def visualize_augmentation():

    # Load the dataset
    train_ds, val_ds, test_ds = get_datasets(
        data_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "processed")
    )

    # Create the augmentation pipeline
    augmenter = get_augmentation_layer()

    # Take one sample image from the training set
    for images, labels in train_ds.take(1):
        sample_image = images[0]
        break

    plt.figure(figsize=(15, 3))

    # Display the original image
    plt.subplot(1, 6, 1)
    plt.imshow(sample_image.numpy().astype("uint8"))
    plt.title("Original")
    plt.axis("off")     # Hide axis for better visualization

    # Generate and display augmented versions
    for i in range(5):
        augmented = augmenter(tf.expand_dims(sample_image, 0), training=True) #tf.expand_dims adds another dimentsion -->(1,224,224,3)
        plt.subplot(1, 6, i + 2)
        plt.imshow(augmented[0].numpy().astype("uint8"))
        plt.title(f"Augmented {i+1}")
        plt.axis("off")
    # Adjust spacing between images
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    visualize_augmentation()