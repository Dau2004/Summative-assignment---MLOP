#!/usr/bin/env python3
"""
Train a new weather classification model with proper class mapping
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("🚀 Starting Weather Classification Model Training")
print(f"TensorFlow version: {tf.__version__}")

# Define paths and constants
DATASET_PATH = '../weather_dataset/'
MODEL_SAVE_PATH = 'model_new_trained.h5'
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    print(f"❌ Dataset not found at {DATASET_PATH}")
    print("Please make sure the weather_dataset folder is in the parent directory")
    exit(1)

# Get class names from folder structure
class_names = [item for item in sorted(os.listdir(DATASET_PATH)) 
               if os.path.isdir(os.path.join(DATASET_PATH, item))]
print(f"✅ Weather classes found: {class_names}")

# Count number of images per class
class_counts = {}
total_images = 0
for class_name in class_names:
    class_dir = os.path.join(DATASET_PATH, class_name)
    count = len([f for f in os.listdir(class_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))])
    class_counts[class_name] = count
    total_images += count

print(f"\n📊 Dataset Statistics:")
for class_name, count in class_counts.items():
    print(f"  {class_name}: {count} images")
print(f"  Total: {total_images} images")

# Create data generators with proper train/validation split
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% for validation
)

# Training generator
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=42,
    shuffle=True
)

# Validation generator  
val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=42,
    shuffle=False
)

print(f"\n🔄 Data Generators Created:")
print(f"  Training samples: {train_generator.samples}")
print(f"  Validation samples: {val_generator.samples}")
print(f"  Class indices: {train_generator.class_indices}")

# Build CNN model
def build_weather_model(input_shape, num_classes):
    """Build a CNN model for weather classification"""
    model = models.Sequential([
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Fourth Conv Block (additional for better feature extraction)
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create model
input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
num_classes = len(class_names)
model = build_weather_model(input_shape, num_classes)

print(f"\n🏗️ Model Architecture:")
model.summary()

# Define callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.0001,
        verbose=1
    )
]

print(f"\n🎯 Starting Training...")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Image size: {IMAGE_SIZE}")

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Save the model
model.save(MODEL_SAVE_PATH)
print(f"\n💾 Model saved as: {MODEL_SAVE_PATH}")

# Evaluate the model
print(f"\n📈 Final Evaluation:")
val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
print(f"  Validation Loss: {val_loss:.4f}")
print(f"  Validation Accuracy: {val_accuracy:.4f}")

# Generate classification report
val_generator.reset()  # Reset generator
predictions = model.predict(val_generator, verbose=0)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes

print(f"\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Print final class mapping for verification
print(f"\n✅ Final Class Mapping (for backend use):")
class_mapping = {v: k for k, v in train_generator.class_indices.items()}
print(f"class_names = {[class_mapping[i] for i in range(num_classes)]}")

print(f"\n🎉 Training Complete!")
print(f"📁 Model saved to: {MODEL_SAVE_PATH}")
print(f"🔄 You can now copy this model to replace the old model.h5")
