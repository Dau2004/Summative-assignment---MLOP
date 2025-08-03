from locust import HttpUser, task, between
import random
import io
from PIL import Image
import numpy as np
import time
import json

class WeatherModelUser(HttpUser):
    wait_time = between(0.1, 0.5)  # Wait 0.1-0.5 seconds between requests
    
    def on_start(self):
        """Generate test images when user starts"""
        self.test_images = self.generate_test_images()
    
    def generate_test_images(self):
        """Generate synthetic test images for different weather conditions"""
        images = []
        weather_types = ['sunny', 'rainy', 'cloudy', 'stormy']
        
        for weather in weather_types:
            # Create synthetic image data
            if weather == 'sunny':
                # Bright yellow/orange image
                img_array = np.random.randint(200, 255, (128, 128, 3), dtype=np.uint8)
                img_array[:, :, 0] = np.random.randint(240, 255, (128, 128))  # High red
                img_array[:, :, 1] = np.random.randint(220, 255, (128, 128))  # High green
                img_array[:, :, 2] = np.random.randint(100, 180, (128, 128))  # Lower blue
            elif weather == 'rainy':
                # Dark blue/gray image
                img_array = np.random.randint(50, 120, (128, 128, 3), dtype=np.uint8)
                img_array[:, :, 2] = np.random.randint(80, 150, (128, 128))  # Higher blue
            elif weather == 'cloudy':
                # Gray image
                gray_val = np.random.randint(120, 180, (128, 128))
                img_array = np.stack([gray_val, gray_val, gray_val], axis=2)
            else:  # stormy
                # Very dark image
                img_array = np.random.randint(20, 80, (128, 128, 3), dtype=np.uint8)
            
            # Convert to PIL Image and then to bytes
            img = Image.fromarray(img_array.astype('uint8'), 'RGB')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            images.append({
                'name': f'{weather}_test.jpg',
                'data': img_bytes.getvalue(),
                'weather': weather
            })
        
        return images
    
    @task(8)
    def predict_weather(self):
        """Main prediction task - 80% of requests"""
        image = random.choice(self.test_images)
        
        files = {
            'image': (image['name'], io.BytesIO(image['data']), 'image/jpeg')
        }
        
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if 'prediction' in result and 'confidence' in result:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def check_status(self):
        """Status check task - 10% of requests"""
        with self.client.get("/status", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def check_model_status(self):
        """Model status check task - 10% of requests"""
        with self.client.get("/model/status", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

class HighLoadUser(HttpUser):
    """High-intensity user for stress testing"""
    wait_time = between(0.01, 0.1)  # Very short wait times
    
    def on_start(self):
        self.test_images = self.generate_test_images()
    
    def generate_test_images(self):
        """Generate smaller test images for faster processing"""
        images = []
        for i in range(3):  # Only 3 images to reduce memory
            img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            img = Image.fromarray(img_array, 'RGB')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG', quality=70)  # Lower quality for speed
            img_bytes.seek(0)
            
            images.append({
                'name': f'stress_test_{i}.jpg',
                'data': img_bytes.getvalue()
            })
        
        return images
    
    @task
    def rapid_predict(self):
        """Rapid prediction requests for stress testing"""
        image = random.choice(self.test_images)
        
        files = {
            'image': (image['name'], io.BytesIO(image['data']), 'image/jpeg')
        }
        
        start_time = time.time()
        with self.client.post("/predict", files=files, catch_response=True) as response:
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            if response.status_code == 200:
                if response_time > 5000:  # 5 second timeout
                    response.failure(f"Response too slow: {response_time:.0f}ms")
                else:
                    response.success()
            else:
                response.failure(f"HTTP {response.status_code}")