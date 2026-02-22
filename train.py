import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
DATA_DIR = "../data" 

def build_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Rescaling(1./255),
        
        # Convolutional Layers (Feature Extraction)
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Dense Layers (Decision Making)
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid') # 0 = Healthy, 1 = Failure
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Load Data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=123,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=123,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE)

    # Train
    model = build_model()
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # Save
    os.makedirs('../models', exist_ok=True)
    model.save('../models/crop_model.h5')
    print("Success: Model saved in ../models/crop_model.h5")
