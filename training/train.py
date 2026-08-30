import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import config
from preprocessing.data_load import get_datasets
from models.efficientnet_model import build_model,compile_model
#Loading values from config file
train_ds, val_ds, test_ds=get_datasets(
    data_dir=config.DATA_DIR,
    image_size=config.Image_Size,
    batch_size=config.Batch_Size
)
model, base_model=build_model(input_shape=(*config.Image_Size,3)) #224,224,3 size
model=compile_model(model,learning_rate=config.Phase1_LR)
#Callbacks
checkpoint_cb=ModelCheckpoint(
    filepath=config.ModelPath_Best,
    monitor='val_accuracy', #Checks Accuracy during Epoch
    save_best_only=True,    #If Accuracy is higher, saves it after overwriting previous
    verbose=1
)
early_stop_cb=EarlyStopping(
    monitor='val_loss',  #Stops training if validation loss does not improve, based on patience
    patience=3,
    restore_best_weights=True,
    verbose=1
)
#Train Phase 1 (Frozen)
#history returns info of val_loss,val_accuracy,loss,accuracy for each epoch for graphs
history=model.fit(
    train_ds,  #Provides Augmented Training Data with Shuffling
    validation_data=val_ds,
    epochs=config.Phase1_Epochs,  #Set to 10 (utils>config.py)
    shuffle=False, #Shuffling is already done in get_datasets()
    callbacks=[checkpoint_cb,early_stop_cb]
)
model.save(config.ModelPath_Final)
print("Phase 1 training complete. Best and final models saved.")

def unfreeze_for_finetuning(base_model, fine_tune_at):
    # Unfreeze the base model for Phase 2 training
    base_model.trainable = True

    # Keep early layers frozen
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    return base_model
# Unfreeze selected layers for fine-tuning
base_model = unfreeze_for_finetuning(
    base_model,
    config.FINE_TUNE_AT_LAYER
)

# Recompile after changing trainable layers
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=config.Phase2_LR
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
# Phase 2 callbacks
checkpoint_cb_phase2 = ModelCheckpoint(
    filepath=config.ModelPath_Best,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop_cb_phase2 = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)
print("PHASE 2:Fine-Tuning")
# Phase 2 Training (Unfrozen)
history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=config.Phase1_Epochs,
    epochs=config.Phase1_Epochs + config.Phase2_Epochs,
    shuffle=False,
    callbacks=[
        checkpoint_cb_phase2,
        early_stop_cb_phase2
    ]
)
# Save the fully fine-tuned model
model.save(config.ModelPath_Final)
print("\nPhase 2 training complete. Best and final models saved.")

#Visualization of training and validation loss and accuracy by Saving History
from utils.visualization import plot_training_history
plot_training_history(
    history_phase1=history,          # Phase 1's history object
    history_phase2=history_phase2,   # Phase 2's history object
    save_path=config.SAVE_DIR + "/training_curves.png"
)