#!/usr/bin/env python3
"""
Load Test Runner for Weather Prediction Model
Runs Locust tests with different container configurations and records performance metrics
"""

import subprocess
import time
import json
import requests
import csv
import os
from datetime import datetime
import threading
import signal
import sys

class LoadTestRunner:
    def __init__(self):
        self.results = []
        self.test_running = False
        
    def check_containers_health(self, ports):
        """Check if all containers are healthy"""
        healthy_containers = 0
        for port in ports:
            try:
                response = requests.get(f"http://localhost:{port}/status", timeout=5)
                if response.status_code == 200:
                    healthy_containers += 1
            except:
                pass
        return healthy_containers
    
    def run_docker_containers(self, num_containers):
        """Start specified number of Docker containers"""
        print(f"🚀 Starting {num_containers} Docker containers...")
        
        if num_containers == 1:
            # Single container
            cmd = ["docker-compose", "up", "-d", "weather-api-1"]
            ports = [8001]
        elif num_containers == 2:
            # Two containers
            cmd = ["docker-compose", "up", "-d", "weather-api-1", "weather-api-2"]
            ports = [8001, 8002]
        elif num_containers == 3:
            # Three containers with nginx load balancer
            cmd = ["docker-compose", "up", "-d"]
            ports = [8001, 8002, 8003]
        else:
            raise ValueError("Supported container counts: 1, 2, 3")
        
        # Stop any existing containers
        subprocess.run(["docker-compose", "down"], capture_output=True)
        
        # Start containers
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to start containers: {result.stderr}")
            return False, []
        
        # Wait for containers to be healthy
        print("⏳ Waiting for containers to be healthy...")
        max_wait = 60  # 60 seconds max wait
        wait_time = 0
        
        while wait_time < max_wait:
            healthy = self.check_containers_health(ports)
            if healthy == len(ports):
                print(f"✅ All {num_containers} containers are healthy!")
                return True, ports
            
            print(f"⏳ {healthy}/{len(ports)} containers healthy, waiting...")
            time.sleep(5)
            wait_time += 5
        
        print(f"❌ Timeout waiting for containers to be healthy")
        return False, ports
    
    def run_locust_test(self, target_url, users, spawn_rate, duration, test_name):
        """Run Locust load test"""
        print(f"🔥 Running Locust test: {test_name}")
        print(f"   Target: {target_url}")
        print(f"   Users: {users}, Spawn Rate: {spawn_rate}/s, Duration: {duration}s")
        
        # Create results directory
        os.makedirs("load_test_results", exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"load_test_results/{test_name}_{timestamp}.csv"
        
        # Run Locust in headless mode
        cmd = [
            "locust",
            "-f", "locustfile.py",
            "--host", target_url,
            "--users", str(users),
            "--spawn-rate", str(spawn_rate),
            "--run-time", f"{duration}s",
            "--headless",
            "--csv", csv_file.replace('.csv', ''),
            "--html", csv_file.replace('.csv', '.html')
        ]
        
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 30)
            end_time = time.time()
            
            # Parse results
            stats = self.parse_locust_results(csv_file + "_stats.csv", result.stdout)
            stats.update({
                'test_name': test_name,
                'target_url': target_url,
                'users': users,
                'spawn_rate': spawn_rate,
                'duration': duration,
                'actual_duration': end_time - start_time,
                'timestamp': datetime.now().isoformat()
            })
            
            return stats
            
        except subprocess.TimeoutExpired:
            print(f"⚠️  Test timed out after {duration + 30} seconds")
            return None
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return None
    
    def parse_locust_results(self, csv_file, stdout):
        """Parse Locust CSV results"""
        stats = {
            'total_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0,
            'min_response_time': 0,
            'max_response_time': 0,
            'rps': 0,
            'failure_rate': 0
        }
        
        try:
            if os.path.exists(csv_file):
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row['Name'] == 'Aggregated':
                            stats.update({
                                'total_requests': int(row['Request Count']),
                                'failed_requests': int(row['Failure Count']),
                                'avg_response_time': float(row['Average Response Time']),
                                'min_response_time': float(row['Min Response Time']),
                                'max_response_time': float(row['Max Response Time']),
                                'rps': float(row['Requests/s']),
                                'failure_rate': float(row['Failure Count']) / max(1, int(row['Request Count'])) * 100
                            })
                            break
        except Exception as e:
            print(f"⚠️  Could not parse CSV results: {e}")
            # Try to extract from stdout
            lines = stdout.split('\n')
            for line in lines:
                if 'requests/s' in line.lower():
                    # Extract basic stats from stdout
                    pass
        
        return stats
    
    def run_comprehensive_test(self):
        """Run comprehensive load tests with different container configurations"""
        print("🎯 Starting Comprehensive Load Test Suite")
        print("=" * 60)
        
        # Test configurations
        test_configs = [
            # Container count, users, spawn_rate, duration
            (1, 10, 2, 60),    # Light load, 1 container
            (1, 50, 5, 60),    # Medium load, 1 container
            (1, 100, 10, 60),  # Heavy load, 1 container
            (2, 50, 5, 60),    # Medium load, 2 containers
            (2, 100, 10, 60),  # Heavy load, 2 containers
            (2, 200, 20, 60),  # Very heavy load, 2 containers
            (3, 100, 10, 60),  # Heavy load, 3 containers
            (3, 200, 20, 60),  # Very heavy load, 3 containers
            (3, 500, 50, 60),  # Extreme load, 3 containers
        ]
        
        all_results = []
        
        for containers, users, spawn_rate, duration in test_configs:
            print(f"\n📊 Test Configuration: {containers} containers, {users} users")
            print("-" * 50)
            
            # Start containers
            success, ports = self.run_docker_containers(containers)
            if not success:
                print(f"❌ Skipping test due to container startup failure")
                continue
            
            # Determine target URL
            if containers == 3:
                target_url = "http://localhost:8000"  # Nginx load balancer
            else:
                target_url = f"http://localhost:{ports[0]}"  # Direct to first container
            
            # Wait a bit for containers to stabilize
            time.sleep(10)
            
            # Run test
            test_name = f"containers_{containers}_users_{users}"
            results = self.run_locust_test(target_url, users, spawn_rate, duration, test_name)
            
            if results:
                results['containers'] = containers
                all_results.append(results)
                
                # Print immediate results
                print(f"✅ Test completed:")
                print(f"   Total Requests: {results['total_requests']}")
                print(f"   Failed Requests: {results['failed_requests']}")
                print(f"   Failure Rate: {results['failure_rate']:.2f}%")
                print(f"   Avg Response Time: {results['avg_response_time']:.2f}ms")
                print(f"   Requests/sec: {results['rps']:.2f}")
            
            # Stop containers before next test
            subprocess.run(["docker-compose", "down"], capture_output=True)
            time.sleep(5)
        
        # Save comprehensive results
        self.save_comprehensive_results(all_results)
        self.generate_performance_report(all_results)
        
        return all_results
    
    def save_comprehensive_results(self, results):
        """Save all test results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"load_test_results/comprehensive_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Comprehensive results saved to: {filename}")
    
    def generate_performance_report(self, results):
        """Generate performance analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"load_test_results/performance_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Weather Model Load Test Performance Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Test Summary\n\n")
            f.write("| Containers | Users | RPS | Avg Response (ms) | Failure Rate (%) | Total Requests |\n")
            f.write("|------------|-------|-----|-------------------|------------------|----------------|\n")
            
            for result in results:
                f.write(f"| {result['containers']} | {result['users']} | "
                       f"{result['rps']:.1f} | {result['avg_response_time']:.1f} | "
                       f"{result['failure_rate']:.2f} | {result['total_requests']} |\n")
            
            f.write("\n## Performance Analysis\n\n")
            
            # Analyze scaling efficiency
            container_groups = {}
            for result in results:
                containers = result['containers']
                if containers not in container_groups:
                    container_groups[containers] = []
                container_groups[containers].append(result)
            
            f.write("### Scaling Efficiency\n\n")
            for containers in sorted(container_groups.keys()):
                f.write(f"**{containers} Container(s):**\n")
                group_results = container_groups[containers]
                
                for result in group_results:
                    f.write(f"- {result['users']} users: {result['rps']:.1f} RPS, "
                           f"{result['avg_response_time']:.1f}ms avg response\n")
                f.write("\n")
            
            f.write("### Key Findings\n\n")
            
            # Find best performing configuration
            best_rps = max(results, key=lambda x: x['rps'])
            f.write(f"- **Highest RPS**: {best_rps['rps']:.1f} with {best_rps['containers']} containers and {best_rps['users']} users\n")
            
            # Find lowest latency
            best_latency = min(results, key=lambda x: x['avg_response_time'])
            f.write(f"- **Lowest Latency**: {best_latency['avg_response_time']:.1f}ms with {best_latency['containers']} containers and {best_latency['users']} users\n")
            
            # Find most reliable
            most_reliable = min(results, key=lambda x: x['failure_rate'])
            f.write(f"- **Most Reliable**: {most_reliable['failure_rate']:.2f}% failure rate with {most_reliable['containers']} containers and {most_reliable['users']} users\n")
        
        print(f"📊 Performance report generated: {report_file}")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Stopping load tests...")
    subprocess.run(["docker-compose", "down"], capture_output=True)
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    runner = LoadTestRunner()
    
    print("🔥 Weather Model Load Test Runner")
    print("=" * 40)
    print("This will test the model with different Docker container configurations")
    print("Press Ctrl+C to stop at any time")
    print()
    
    try:
        results = runner.run_comprehensive_test()
        print("\n🎉 All load tests completed successfully!")
        print(f"📊 {len(results)} test configurations executed")
        
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
    finally:
        # Cleanup
        subprocess.run(["docker-compose", "down"], capture_output=True)
        print("🧹 Cleanup completed")