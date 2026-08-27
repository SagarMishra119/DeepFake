import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from utils import config
from preprocessing.data_load import get_datasets

# Loading the best model saved during training
model = tf.keras.models.load_model(config.ModelPath_Best)

#_,_, ignores values of train_ds and val_ds as only test_ds is needed
_, _, test_ds = get_datasets( 
    data_dir=config.DATA_DIR, #DATA_DIR is the path to the dataset directory
    image_size=config.Image_Size,
    batch_size=config.Batch_Size
)

y_true = []        # Stores the actual labels from the test dataset
y_pred_probs = []  # Stores prediction probabilities from the model

for images, labels in test_ds:
    preds = model.predict(images, verbose=0) #Verbose set to 0 to suppress output during prediction 
    y_true.extend(labels.numpy().flatten())
    y_pred_probs.extend(preds.flatten())

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)

# Convert probabilities into class labels (0 or 1)
y_pred = (y_pred_probs >= 0.5).astype(int)   # Final predicted labels

acc = accuracy_score(y_true, y_pred)              # Overall prediction accuracy
precision = precision_score(y_true, y_pred)       # Correct positive predictions out of predicted positives
recall = recall_score(y_true, y_pred)             # Correct positive predictions out of actual positives
f1 = f1_score(y_true, y_pred)                     # Balance between precision and recall
cm = confusion_matrix(y_true, y_pred)             # Shows correct and incorrect predictions

print(f"Test Accuracy:  {acc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall:    {recall:.4f}")
print(f"Test F1-score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nFull classification report:")
print(classification_report(y_true, y_pred, target_names=["fake", "real"]))

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["fake", "real"], yticklabels=["fake", "real"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Test Set")

plt.tight_layout()
plt.savefig("evaluation/confusion_matrix.png")
plt.show()