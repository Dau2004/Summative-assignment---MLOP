#!/usr/bin/env python3
"""
Quick Load Test Demo
Demonstrates flood of requests to the model with real-time metrics
"""

import asyncio
import aiohttp
import time
import json
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from datetime import datetime
import threading
import queue

class QuickLoadTester:
    def __init__(self):
        self.results = []
        self.response_times = []
        self.success_count = 0
        self.error_count = 0
        self.start_time = None
        
    def generate_test_image(self, weather_type='random'):
        """Generate a synthetic test image"""
        if weather_type == 'random':
            weather_type = np.random.choice(['sunny', 'rainy', 'cloudy', 'stormy'])
        
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
        image_data = self.generate_test_image()
        
        data = aiohttp.FormData()
        data.add_field('image', image_data, filename=f'test_{request_id}.jpg', content_type='image/jpeg')
        
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
                        'confidence': result.get('confidence', 0)
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
    
    async def flood_requests(self, url, num_requests, concurrent_requests=10):
        """Send a flood of requests to the model"""
        print(f"🌊 Flooding {url} with {num_requests} requests ({concurrent_requests} concurrent)")
        
        self.start_time = time.time()
        
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
    
    def print_real_time_stats(self):
        """Print real-time statistics during the test"""
        while True:
            if self.start_time and self.results:
                elapsed = time.time() - self.start_time
                total_requests = len(self.results)
                rps = total_requests / elapsed if elapsed > 0 else 0
                
                if self.response_times:
                    avg_response = np.mean(self.response_times)
                    p95_response = np.percentile(self.response_times, 95)
                else:
                    avg_response = 0
                    p95_response = 0
                
                success_rate = (self.success_count / total_requests * 100) if total_requests > 0 else 0
                
                print(f"\r⚡ Live Stats: {total_requests} reqs | {rps:.1f} RPS | "
                      f"{avg_response:.0f}ms avg | {p95_response:.0f}ms p95 | "
                      f"{success_rate:.1f}% success", end="", flush=True)
            
            time.sleep(1)
    
    def analyze_results(self):
        """Analyze and display test results"""
        if not self.results:
            print("❌ No results to analyze")
            return
        
        total_time = time.time() - self.start_time
        total_requests = len(self.results)
        
        print(f"\n\n📊 Load Test Results")
        print("=" * 50)
        print(f"Total Requests: {total_requests}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Requests/sec: {total_requests / total_time:.2f}")
        print(f"Successful: {self.success_count}")
        print(f"Failed: {self.error_count}")
        print(f"Success Rate: {self.success_count / total_requests * 100:.2f}%")
        
        if self.response_times:
            print(f"\n⏱️  Response Time Statistics:")
            print(f"Average: {np.mean(self.response_times):.2f}ms")
            print(f"Median: {np.median(self.response_times):.2f}ms")
            print(f"95th Percentile: {np.percentile(self.response_times, 95):.2f}ms")
            print(f"99th Percentile: {np.percentile(self.response_times, 99):.2f}ms")
            print(f"Min: {np.min(self.response_times):.2f}ms")
            print(f"Max: {np.max(self.response_times):.2f}ms")
        
        # Analyze predictions
        predictions = [r.get('prediction', 'unknown') for r in self.results if r.get('status') == 'success']
        if predictions:
            from collections import Counter
            pred_counts = Counter(predictions)
            print(f"\n🌤️  Prediction Distribution:")
            for pred, count in pred_counts.most_common():
                percentage = count / len(predictions) * 100
                print(f"  {pred}: {count} ({percentage:.1f}%)")
    
    def plot_response_times(self):
        """Plot response time distribution"""
        if not self.response_times:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Response time histogram
        plt.subplot(2, 2, 1)
        plt.hist(self.response_times, bins=50, alpha=0.7, color='blue')
        plt.title('Response Time Distribution')
        plt.xlabel('Response Time (ms)')
        plt.ylabel('Frequency')
        
        # Response time over time
        plt.subplot(2, 2, 2)
        plt.plot(self.response_times, alpha=0.7, color='green')
        plt.title('Response Time Over Requests')
        plt.xlabel('Request Number')
        plt.ylabel('Response Time (ms)')
        
        # Box plot
        plt.subplot(2, 2, 3)
        plt.boxplot(self.response_times)
        plt.title('Response Time Box Plot')
        plt.ylabel('Response Time (ms)')
        
        # Cumulative response time
        plt.subplot(2, 2, 4)
        sorted_times = np.sort(self.response_times)
        percentiles = np.arange(1, len(sorted_times) + 1) / len(sorted_times) * 100
        plt.plot(sorted_times, percentiles, color='red')
        plt.title('Response Time Percentiles')
        plt.xlabel('Response Time (ms)')
        plt.ylabel('Percentile')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"load_test_results/response_times_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📈 Response time plot saved: {filename}")
        
        plt.show()
    
    def save_results(self):
        """Save detailed results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"load_test_results/quick_test_{timestamp}.json"
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_requests': len(self.results),
            'successful_requests': self.success_count,
            'failed_requests': self.error_count,
            'total_time': time.time() - self.start_time if self.start_time else 0,
            'requests_per_second': len(self.results) / (time.time() - self.start_time) if self.start_time else 0,
            'response_time_stats': {
                'average': float(np.mean(self.response_times)) if self.response_times else 0,
                'median': float(np.median(self.response_times)) if self.response_times else 0,
                'p95': float(np.percentile(self.response_times, 95)) if self.response_times else 0,
                'p99': float(np.percentile(self.response_times, 99)) if self.response_times else 0,
                'min': float(np.min(self.response_times)) if self.response_times else 0,
                'max': float(np.max(self.response_times)) if self.response_times else 0
            },
            'detailed_results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Detailed results saved: {filename}")

async def main():
    """Main function to run the quick load test"""
    print("🚀 Quick Load Test for Weather Prediction Model")
    print("=" * 50)
    
    # Test configurations
    test_url = "http://localhost:8000"  # Change this to your model's URL
    num_requests = 100  # Number of requests to send
    concurrent_requests = 20  # Number of concurrent requests
    
    print(f"Target URL: {test_url}")
    print(f"Total Requests: {num_requests}")
    print(f"Concurrent Requests: {concurrent_requests}")
    print()
    
    # Create tester
    tester = QuickLoadTester()
    
    # Start real-time stats in background
    stats_thread = threading.Thread(target=tester.print_real_time_stats, daemon=True)
    stats_thread.start()
    
    try:
        # Run the flood test
        total_time = await tester.flood_requests(test_url, num_requests, concurrent_requests)
        
        # Analyze results
        tester.analyze_results()
        
        # Create visualizations
        tester.plot_response_times()
        
        # Save results
        tester.save_results()
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    # Create results directory
    import os
    os.makedirs("load_test_results", exist_ok=True)
    
    # Run the test
    asyncio.run(main())