#!/usr/bin/env python3
"""
Test script to verify the enhanced prediction is working in the server context
"""
import os
import sys
sys.path.append('.')

# Import the enhanced prediction logic
from main import enhanced_predict, model, class_names
import numpy as np
from PIL import Image
import tensorflow as tf

def test_rain_image():
    """Test with a rain image to see if enhanced prediction works"""
    # Load a rain image from uploaded_data
    rain_images_path = "uploaded_data/Rain"
    if os.path.exists(rain_images_path):
        rain_files = [f for f in os.listdir(rain_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if rain_files:
            image_path = os.path.join(rain_images_path, rain_files[0])
            print(f"🧪 Testing enhanced prediction with: {image_path}")
            
            # Load and preprocess image
            image_pil = Image.open(image_path).convert('RGB')
            image_pil = image_pil.resize((128, 128))
            image_array = np.array(image_pil) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            # Test enhanced prediction
            if model:
                predicted_class, confidence, enhanced_predictions = enhanced_predict(model, image_array, class_names)
                
                # Also get raw prediction for comparison
                raw_predictions = model.predict(image_array, verbose=0)[0]
                
                print(f"\n📊 ENHANCED PREDICTION TEST RESULTS:")
                print(f"Image: {rain_files[0]} (should be Rain)")
                print(f"Predicted: {predicted_class} ({confidence:.3f})")
                print(f"Raw predictions: {[f'{class_names[i]}:{raw_predictions[i]:.3f}' for i in range(len(class_names))]}")
                print(f"Enhanced predictions: {[f'{class_names[i]}:{enhanced_predictions[i]:.3f}' for i in range(len(class_names))]}")
                
                # Check if Rain prediction improved
                raw_rain_conf = raw_predictions[1]  # Rain is at index 1
                enhanced_rain_conf = enhanced_predictions[1]
                print(f"\n🌧️ Rain confidence: Raw={raw_rain_conf:.3f} → Enhanced={enhanced_rain_conf:.3f}")
                
                if enhanced_rain_conf > raw_rain_conf:
                    print("✅ Enhancement IMPROVED Rain detection!")
                else:
                    print("❌ Enhancement did not improve Rain detection")
                    
                return predicted_class == "Rain"
            else:
                print("❌ Model not loaded!")
                return False
    else:
        print(f"❌ Rain images directory not found: {rain_images_path}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Enhanced Prediction Server Logic")
    success = test_rain_image()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: Enhanced prediction test")
