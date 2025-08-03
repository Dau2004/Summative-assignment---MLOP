#!/usr/bin/env python3
"""
Demo Load Test - Shows flood of requests to the weather model
"""

import asyncio
import aiohttp
import time
import json
import numpy as np
from PIL import Image
import io
import threading
from datetime import datetime

class LoadTestDemo:
    def __init__(self):
        self.results = []
        self.response_times = []
        self.success_count = 0
        self.error_count = 0
        self.start_time = None
        
    def generate_test_image(self, weather_type='sunny'):
        """Generate a synthetic test image"""
        # Create synthetic image based on weather type
        if weather_type == 'sunny':
            img_array = np.random.randint(200, 255, (128, 128, 3), dtype=np.uint8)
            img_array[:, :, 0] = np.random.randint(240, 255, (128, 128))  # High red
            img_array[:, :, 1] = np.random.randint(220, 255, (128, 128))  # High green
        elif weather_type == 'rainy':
            img_array = np.random.randint(50, 120, (128, 128, 3), dtype=np.uint8)
            img_array[:, :, 2] = np.random.randint(80, 150, (128, 128))  # Higher blue
        elif weather_type == 'cloudy':
            gray_val = np.random.randint(120, 180, (128, 128))
            img_array = np.stack([gray_val, gray_val, gray_val], axis=2)
        else:  # stormy
            img_array = np.random.randint(20, 80, (128, 128, 3), dtype=np.uint8)
        
        # Convert to bytes
        img = Image.fromarray(img_array.astype('uint8'), 'RGB')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG', quality=85)
        img_bytes.seek(0)
        
        return img_bytes.getvalue()
    
    async def send_prediction_request(self, session, url, request_id):
        """Send a single prediction request"""
        weather_types = ['sunny', 'rainy', 'cloudy', 'stormy']
        weather_type = np.random.choice(weather_types)
        image_data = self.generate_test_image(weather_type)
        
        data = aiohttp.FormData()
        data.add_field('image', image_data, filename=f'test_{request_id}_{weather_type}.jpg', content_type='image/jpeg')
        
        start_time = time.time()
        try:
            async with session.post(f"{url}/predict", data=data) as response:
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                
                if response.status == 200:
                    result = await response.json()
                    self.success_count += 1
                    self.response_times.append(response_time)
                    return {
                        'request_id': request_id,
                        'status': 'success',
                        'response_time': response_time,
                        'prediction': result.get('prediction', 'unknown'),
                        'confidence': result.get('confidence', 0),
                        'actual_weather': weather_type
                    }
                else:
                    self.error_count += 1
                    return {
                        'request_id': request_id,
                        'status': 'error',
                        'response_time': response_time,
                        'error': f"HTTP {response.status}"
                    }
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.error_count += 1
            return {
                'request_id': request_id,
                'status': 'error',
                'response_time': response_time,
                'error': str(e)
            }
    
    def print_live_stats(self):
        """Print live statistics during the test"""
        while self.start_time and (time.time() - self.start_time) < 30:  # Run for 30 seconds max
            if self.results:
                elapsed = time.time() - self.start_time
                total_requests = len(self.results)
                rps = total_requests / elapsed if elapsed > 0 else 0
                
                if self.response_times:
                    avg_response = np.mean(self.response_times)
                    p95_response = np.percentile(self.response_times, 95) if len(self.response_times) > 1 else avg_response
                else:
                    avg_response = 0
                    p95_response = 0
                
                success_rate = (self.success_count / total_requests * 100) if total_requests > 0 else 0
                
                print(f"\r🔥 FLOOD TEST: {total_requests:3d} reqs | {rps:5.1f} RPS | "
                      f"{avg_response:5.0f}ms avg | {p95_response:5.0f}ms p95 | "
                      f"{success_rate:5.1f}% success | {self.error_count} errors", 
                      end="", flush=True)
            
            time.sleep(0.5)
    
    async def flood_test(self, url, num_requests=50, concurrent_requests=10):
        """Send a flood of requests"""
        print(f"🌊 STARTING FLOOD TEST")
        print(f"Target: {url}")
        print(f"Requests: {num_requests} (concurrent: {concurrent_requests})")
        print("=" * 80)
        
        self.start_time = time.time()
        
        # Start live stats in background
        stats_thread = threading.Thread(target=self.print_live_stats, daemon=True)
        stats_thread.start()
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def bounded_request(session, request_id):
            async with semaphore:
                return await self.send_prediction_request(session, url, request_id)
        
        # Create session with connection pooling
        connector = aiohttp.TCPConnector(limit=concurrent_requests * 2)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create all tasks
            tasks = [bounded_request(session, i) for i in range(num_requests)]
            
            # Execute tasks and collect results
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and add to results
            for result in results:
                if isinstance(result, dict):
                    self.results.append(result)
        
        total_time = time.time() - self.start_time
        return total_time
    
    def analyze_results(self):
        """Analyze and display test results"""
        if not self.results:
            print("\n❌ No results to analyze")
            return
        
        total_time = time.time() - self.start_time
        total_requests = len(self.results)
        
        print(f"\n\n📊 FLOOD TEST RESULTS")
        print("=" * 60)
        print(f"🎯 Total Requests: {total_requests}")
        print(f"⏱️  Total Time: {total_time:.2f}s")
        print(f"🚀 Requests/sec: {total_requests / total_time:.2f}")
        print(f"✅ Successful: {self.success_count}")
        print(f"❌ Failed: {self.error_count}")
        print(f"📈 Success Rate: {self.success_count / total_requests * 100:.2f}%")
        
        if self.response_times:
            print(f"\n⚡ RESPONSE TIME ANALYSIS:")
            print(f"   Average: {np.mean(self.response_times):.0f}ms")
            print(f"   Median: {np.median(self.response_times):.0f}ms")
            print(f"   95th Percentile: {np.percentile(self.response_times, 95):.0f}ms")
            print(f"   99th Percentile: {np.percentile(self.response_times, 99):.0f}ms")
            print(f"   Min: {np.min(self.response_times):.0f}ms")
            print(f"   Max: {np.max(self.response_times):.0f}ms")
        
        # Analyze predictions vs actual
        successful_results = [r for r in self.results if r.get('status') == 'success']
        if successful_results:
            print(f"\n🌤️  PREDICTION ACCURACY:")
            correct_predictions = 0
            for result in successful_results:
                actual = result.get('actual_weather', '').lower()
                predicted = result.get('prediction', '').lower()
                
                # Simple matching logic
                if (actual == 'sunny' and 'shine' in predicted) or \
                   (actual == 'rainy' and 'rain' in predicted) or \
                   (actual == 'cloudy' and 'cloud' in predicted) or \
                   (actual == 'stormy' and ('rain' in predicted or 'cloud' in predicted)):
                    correct_predictions += 1
            
            accuracy = correct_predictions / len(successful_results) * 100
            print(f"   Model Accuracy: {accuracy:.1f}% ({correct_predictions}/{len(successful_results)})")
            
            # Show prediction distribution
            from collections import Counter
            predictions = [r.get('prediction', 'unknown') for r in successful_results]
            pred_counts = Counter(predictions)
            print(f"   Prediction Distribution:")
            for pred, count in pred_counts.most_common():
                percentage = count / len(predictions) * 100
                print(f"     {pred}: {count} ({percentage:.1f}%)")

async def main():
    """Run the demo load test"""
    print("🔥 WEATHER MODEL FLOOD TEST DEMO")
    print("=" * 50)
    
    # Test configuration
    test_url = "http://localhost:8000"
    num_requests = 50  # Moderate number for demo
    concurrent_requests = 15  # High concurrency to simulate flood
    
    # Create tester
    tester = LoadTestDemo()
    
    try:
        # Check if server is running
        import requests
        response = requests.get(f"{test_url}/status", timeout=5)
        if response.status_code != 200:
            print(f"❌ Server not responding at {test_url}")
            return
        
        print(f"✅ Server is running at {test_url}")
        print(f"🎯 Preparing to flood with {num_requests} requests...")
        print()
        
        # Run the flood test
        total_time = await tester.flood_test(test_url, num_requests, concurrent_requests)
        
        # Analyze results
        tester.analyze_results()
        
        print(f"\n🎉 FLOOD TEST COMPLETED!")
        print(f"📊 The model handled {len(tester.results)} requests in {total_time:.2f} seconds")
        
        # Performance assessment
        rps = len(tester.results) / total_time
        avg_response = np.mean(tester.response_times) if tester.response_times else 0
        
        print(f"\n🏆 PERFORMANCE ASSESSMENT:")
        if rps > 10:
            print(f"   🚀 EXCELLENT: {rps:.1f} RPS - Model handles high load well!")
        elif rps > 5:
            print(f"   ✅ GOOD: {rps:.1f} RPS - Model performs adequately under load")
        else:
            print(f"   ⚠️  NEEDS IMPROVEMENT: {rps:.1f} RPS - Consider optimization")
        
        if avg_response < 500:
            print(f"   ⚡ FAST: {avg_response:.0f}ms average response time")
        elif avg_response < 1000:
            print(f"   ✅ ACCEPTABLE: {avg_response:.0f}ms average response time")
        else:
            print(f"   🐌 SLOW: {avg_response:.0f}ms average response time")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    # Create results directory
    import os
    os.makedirs("load_test_results", exist_ok=True)
    
    # Run the demo
    asyncio.run(main())