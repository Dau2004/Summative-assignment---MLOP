#!/usr/bin/env python3
"""
Simple Locust Demo
Shows how to run a basic Locust test against the weather model
"""

import subprocess
import time
import requests
import os
import signal
import sys

def check_server_status():
    """Check if the weather model server is running"""
    try:
        response = requests.get("http://localhost:8000/status", timeout=5)
        return response.status_code == 200
    except:
        return False

def run_locust_demo():
    """Run a simple Locust demonstration"""
    print("🔥 Simple Locust Load Test Demo")
    print("=" * 40)
    
    # Check if server is running
    if not check_server_status():
        print("❌ Weather model server is not running at http://localhost:8000")
        print("💡 Please start the server first: python3 main.py")
        return
    
    print("✅ Server is running at http://localhost:8000")
    print()
    
    # Run Locust in headless mode for demo
    print("🚀 Starting Locust load test...")
    print("   Users: 20")
    print("   Spawn Rate: 5 users/second")
    print("   Duration: 30 seconds")
    print("   Target: http://localhost:8000")
    print()
    
    # Create results directory
    os.makedirs("load_test_results", exist_ok=True)
    
    # Generate timestamp for unique filenames
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run Locust command
    cmd = [
        "locust",
        "-f", "locustfile.py",
        "--host", "http://localhost:8000",
        "--users", "20",
        "--spawn-rate", "5",
        "--run-time", "30s",
        "--headless",
        "--csv", f"load_test_results/locust_demo_{timestamp}",
        "--html", f"load_test_results/locust_demo_{timestamp}.html"
    ]
    
    try:
        print("⏳ Running test... (30 seconds)")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        
        if result.returncode == 0:
            print("✅ Locust test completed successfully!")
            print()
            
            # Parse and display results from stdout
            lines = result.stdout.split('\n')
            for line in lines:
                if 'requests/s' in line.lower() or 'response time' in line.lower():
                    print(f"📊 {line.strip()}")
            
            # Show generated files
            print(f"\\n📁 Results saved:")
            print(f"   📊 HTML Report: load_test_results/locust_demo_{timestamp}.html")
            print(f"   📈 CSV Data: load_test_results/locust_demo_{timestamp}_stats.csv")
            
        else:
            print(f"❌ Locust test failed:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⚠️  Test timed out (this is normal for the demo)")
    except FileNotFoundError:
        print("❌ Locust is not installed. Install it with: pip3 install locust")
    except Exception as e:
        print(f"❌ Test failed: {e}")

def run_interactive_locust():
    """Run Locust with web interface"""
    print("🌐 Starting Locust Web Interface")
    print("=" * 40)
    
    if not check_server_status():
        print("❌ Weather model server is not running at http://localhost:8000")
        return
    
    print("✅ Server is running")
    print("🚀 Starting Locust web interface...")
    print("📱 Open your browser to: http://localhost:8089")
    print("🎯 Target host: http://localhost:8000")
    print("💡 Suggested settings:")
    print("   - Number of users: 20")
    print("   - Spawn rate: 5")
    print("   - Host: http://localhost:8000")
    print()
    print("Press Ctrl+C to stop")
    print()
    
    try:
        # Run Locust with web interface
        cmd = ["locust", "-f", "locustfile.py", "--host", "http://localhost:8000"]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\\n🛑 Locust stopped by user")
    except FileNotFoundError:
        print("❌ Locust is not installed. Install it with: pip3 install locust")

def main():
    """Main function with menu"""
    print("🔥 Locust Load Testing Demo")
    print("=" * 30)
    print("Choose an option:")
    print("1. Quick headless demo (30 seconds)")
    print("2. Interactive web interface")
    print("3. Exit")
    print()
    
    try:
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            run_locust_demo()
        elif choice == "2":
            run_interactive_locust()
        elif choice == "3":
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice. Please run again.")
            
    except KeyboardInterrupt:
        print("\\n👋 Goodbye!")

if __name__ == "__main__":
    main()