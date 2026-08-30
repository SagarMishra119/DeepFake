import os
Collab=False #Collab used for Training Model Only
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#Pathing 
if Collab:
    DATA_DIR = "/content/drive/MyDrive/DeepFake/processed" #Used later on to recall this directory
    SAVE_DIR = "/content/drive/MyDrive/DeepFake/saved_models"
else:
    DATA_DIR = os.path.join(BASE_DIR, "processed") #Used later on to recall this directory
    SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True) #Checks if the directory exists, else creates it
ModelPath_Best=os.path.join(SAVE_DIR,"best_model.keras")
ModelPath_Final=os.path.join(SAVE_DIR,"final_model.keras")
#Image Data(Size,etc.)
Image_Size=(224,224) #Image Size for Training
Batch_Size=64 #Batch Size for Training
#Hyperparameters
#Phase1 (Frozen)
Phase1_Epochs=10 #Phase 1 Training Epochs 
Phase1_LR=0.001 #Phase 1 Learning Rate
#Phase2(Unfrozen)
Phase2_Epochs=30 #Phase 2 Training Epochs
Phase2_LR=0.00001 #Phase 2 Learning Rate
FINE_TUNE_AT_LAYER = 100
