#!/usr/bin/env python3
"""
Simple Load Test Demo - No Docker Required
"""

import subprocess
import time
import requests
import asyncio
import aiohttp
import numpy as np
from PIL import Image
import io
import threading
import os

def check_server():
    """Check if server is running"""
    try:
        response = requests.get("http://localhost:8000/status", timeout=3)
        return response.status_code == 200
    except:
        return False

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting weather model server...")
    try:
        # Start server in background
        process = subprocess.Popen(
            ["python", "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        for i in range(10):
            if check_server():
                print("✅ Server started successfully!")
                return process
            print(f"⏳ Waiting for server... ({i+1}/10)")
            time.sleep(2)
        
        print("❌ Server failed to start")
        process.terminate()
        return None
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return None

async def flood_test():
    """Run flood test"""
    print("\n🌊 STARTING FLOOD TEST")
    print("=" * 40)
    
    # Generate test image
    def generate_image():
        img_array = np.random.randint(100, 200, (128, 128, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes.getvalue()
    
    # Test parameters
    num_requests = 20
    concurrent = 5
    
    print(f"📊 Sending {num_requests} requests with {concurrent} concurrent connections")
    
    results = []
    start_time = time.time()
    
    async def send_request(session, i):
        image_data = generate_image()
        data = aiohttp.FormData()
        data.add_field('image', image_data, filename=f'test_{i}.jpg', content_type='image/jpeg')
        
        req_start = time.time()
        try:
            async with session.post("http://localhost:8000/predict", data=data) as response:
                response_time = (time.time() - req_start) * 1000
                if response.status == 200:
                    result = await response.json()
                    return {
                        'success': True,
                        'response_time': response_time,
                        'prediction': result.get('prediction', 'unknown')
                    }
                else:
                    return {'success': False, 'response_time': response_time}
        except Exception as e:
            return {'success': False, 'response_time': (time.time() - req_start) * 1000, 'error': str(e)}
    
    # Run requests
    semaphore = asyncio.Semaphore(concurrent)
    
    async def bounded_request(session, i):
        async with semaphore:
            return await send_request(session, i)
    
    connector = aiohttp.TCPConnector(limit=concurrent * 2)
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [bounded_request(session, i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful = [r for r in results if isinstance(r, dict) and r.get('success')]
    failed = len(results) - len(successful)
    
    if successful:
        response_times = [r['response_time'] for r in successful]
        avg_response = np.mean(response_times)
        rps = len(results) / total_time
        
        print(f"\n📊 RESULTS:")
        print(f"   Total Requests: {len(results)}")
        print(f"   Successful: {len(successful)}")
        print(f"   Failed: {failed}")
        print(f"   Success Rate: {len(successful)/len(results)*100:.1f}%")
        print(f"   Requests/sec: {rps:.1f}")
        print(f"   Avg Response Time: {avg_response:.0f}ms")
        
        # Show predictions
        predictions = [r.get('prediction', 'unknown') for r in successful]
        from collections import Counter
        pred_counts = Counter(predictions)
        print(f"   Predictions: {dict(pred_counts)}")
        
        print(f"\n🏆 ASSESSMENT:")
        if rps > 10:
            print(f"   🚀 EXCELLENT: {rps:.1f} RPS")
        elif rps > 5:
            print(f"   ✅ GOOD: {rps:.1f} RPS")
        else:
            print(f"   ⚠️  NEEDS IMPROVEMENT: {rps:.1f} RPS")
    else:
        print("❌ All requests failed")

def main():
    """Main function"""
    print("🔥 SIMPLE WEATHER MODEL LOAD TEST")
    print("=" * 50)
    
    # Check if server is already running
    if check_server():
        print("✅ Server is already running")
        server_process = None
    else:
        # Start server
        server_process = start_server()
        if not server_process:
            return
    
    try:
        # Run flood test
        asyncio.run(flood_test())
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
    finally:
        # Cleanup
        if server_process:
            print("\n🧹 Stopping server...")
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    main()