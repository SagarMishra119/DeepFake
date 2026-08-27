import matplotlib.pyplot as plt

def plot_training_history(history_phase1, history_phase2, save_path=None):
    """
    Combines Phase 1 and Phase 2 history objects into one continuous
    plot of accuracy and loss across the full training run.
    """
    # Merge both training phases into one history
    acc = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
    val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
    loss = history_phase1.history['loss'] + history_phase2.history['loss']
    val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    # Marks where fine-tuning begins
    phase_boundary = len(history_phase1.history['accuracy'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(epochs_range, acc, label='Train Accuracy')
    axes[0].plot(epochs_range, val_acc, label='Validation Accuracy')
    axes[0].axvline(x=phase_boundary, color='gray', linestyle='--', label='Fine-tuning starts')
    axes[0].set_title('Accuracy over epochs')  #Plot title
    axes[0].set_xlabel('Epoch')                #Training epochs
    axes[0].set_ylabel('Accuracy')             #Accuracy values
    axes[0].legend()                           #Display plot labels

    # Loss plot
    axes[1].plot(epochs_range, loss, label='Train Loss')
    axes[1].plot(epochs_range, val_loss, label='Validation Loss')
    axes[1].axvline(x=phase_boundary, color='gray', linestyle='--', label='Fine-tuning starts')
    axes[1].set_title('Loss over epochs')      #Plot title
    axes[1].set_xlabel('Epoch')                #Training epochs
    axes[1].set_ylabel('Loss')                 #Loss values
    axes[1].legend()                           #Display plot labels
    # Adjust spacing between plots
    plt.tight_layout()
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")

    plt.show()