import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime
import os

class ModelRetrainer:
    def __init__(self, model_path='model.h5'):
        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
    def load_new_data(self, data_dir):
        self.data_dir = data_dir
        
        # Data generator with augmentation
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2  # 20% for validation
        )
        
        # Load training data
        self.train_generator = self.train_datagen.flow_from_directory(
            self.data_dir,
            target_size=(128, 128),
            batch_size=32,
            class_mode='categorical',
            subset='training',
            seed=42
        )
        
        # Load validation data
        self.val_generator = self.train_datagen.flow_from_directory(
            self.data_dir,
            target_size=(128, 128),
            batch_size=32,
            class_mode='categorical',
            subset='validation',
            seed=42
        )
        
    def retrain(self, epochs=10):
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]
        
        # Retrain model
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks
        )
        
        # Save retrained model
        self.model.save('model.h5')
        
        return history