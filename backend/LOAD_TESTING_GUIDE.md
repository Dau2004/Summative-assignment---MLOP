# Weather Model Load Testing Suite

This comprehensive load testing suite demonstrates how to simulate flood requests to the weather prediction model and analyze performance with different Docker container configurations.

## 🎯 What We've Built

### 1. **Locust Load Testing Framework** (`locustfile.py`)
- **WeatherModelUser**: Simulates normal user behavior with realistic wait times
- **HighLoadUser**: Simulates high-intensity stress testing
- Generates synthetic weather images for testing
- Supports different request patterns and intensities

### 2. **Docker Container Setup**
- **Dockerfile**: Containerizes the weather prediction API
- **docker-compose.yml**: Orchestrates multiple containers with load balancing
- **nginx.conf**: Load balancer configuration for distributing requests

### 3. **Load Testing Scripts**

#### **Quick Demo** (`demo_load_test.py`)
```bash
python3 demo_load_test.py
```
- **Purpose**: Immediate demonstration of flood testing
- **Features**: 
  - Real-time statistics display
  - 50 requests with 15 concurrent connections
  - Live performance metrics
  - Prediction accuracy analysis

#### **Comprehensive Test Runner** (`load_test_runner.py`)
```bash
python3 load_test_runner.py
```
- **Purpose**: Full-scale testing with multiple container configurations
- **Test Matrix**:
  - 1 Container: 10, 50, 100 users
  - 2 Containers: 50, 100, 200 users  
  - 3 Containers: 100, 200, 500 users
- **Outputs**: Detailed performance reports and CSV data

#### **Container Performance Demo** (`container_performance_demo.py`)
```bash
python3 container_performance_demo.py
```
- **Purpose**: Shows scaling behavior with different container counts
- **Features**: Automated Docker container management
- **Analysis**: Performance comparison and scaling efficiency

### 4. **Easy-to-Use Runner** (`run_load_tests.sh`)
```bash
./run_load_tests.sh
```
Interactive menu with options:
1. Quick Demo (100 requests, real-time visualization)
2. Comprehensive Test (Multiple container configurations)
3. Single Container Test (Locust Web UI)
4. Build Docker Images Only

## 🚀 Quick Start

### Prerequisites
```bash
# Install required packages
pip3 install locust aiohttp matplotlib requests

# Ensure Docker is running
docker --version
```

### Run the Demo
```bash
cd backend
./run_load_tests.sh
# Choose option 1 for quick demo
```

## 📊 Performance Results Example

From our demonstration run:

```
🔥 WEATHER MODEL FLOOD TEST DEMO
==================================================
✅ Server is running at http://localhost:8000
🎯 Preparing to flood with 50 requests...

📊 FLOOD TEST RESULTS
============================================================
🎯 Total Requests: 50
⏱️  Total Time: 2.70s
🚀 Requests/sec: 18.52
✅ Successful: 50
❌ Failed: 0
📈 Success Rate: 100.00%

⚡ RESPONSE TIME ANALYSIS:
   Average: 697ms
   Median: 676ms
   95th Percentile: 944ms
   99th Percentile: 1224ms
   Min: 405ms
   Max: 1428ms

🏆 PERFORMANCE ASSESSMENT:
   🚀 EXCELLENT: 18.5 RPS - Model handles high load well!
   ✅ ACCEPTABLE: 697ms average response time
```

## 🔧 Key Features

### **Synthetic Image Generation**
- Creates realistic weather images for testing
- Supports sunny, rainy, cloudy, and stormy conditions
- Optimized for consistent load testing

### **Real-Time Monitoring**
- Live statistics during flood tests
- Response time percentiles (P95, P99)
- Success/failure rates
- Requests per second (RPS)

### **Container Scaling Analysis**
- Automated Docker container orchestration
- Performance comparison across configurations
- Scaling efficiency metrics
- Load balancer integration

### **Comprehensive Reporting**
- JSON results with detailed metrics
- Markdown performance reports
- CSV data for further analysis
- Visual response time distributions

## 📈 Scaling Behavior

The load testing suite demonstrates:

1. **Single Container**: Baseline performance
2. **Multiple Containers**: Horizontal scaling benefits
3. **Load Balancer**: Request distribution efficiency
4. **Resource Utilization**: Memory and CPU impact

## 🎛️ Configuration Options

### Test Parameters
- **Number of users**: 10-500 concurrent users
- **Spawn rate**: 2-50 users per second
- **Test duration**: 60+ seconds
- **Request patterns**: Normal vs. stress testing

### Container Configurations
- **1 Container**: Direct connection testing
- **2 Containers**: Basic load distribution
- **3+ Containers**: Full load balancing with Nginx

## 📁 Output Files

All results are saved in `load_test_results/`:
- `comprehensive_results_TIMESTAMP.json`: Full test data
- `performance_report_TIMESTAMP.md`: Analysis report
- `container_performance_TIMESTAMP.json`: Scaling analysis
- `response_times_TIMESTAMP.png`: Visual charts

## 🛠️ Customization

### Modify Test Parameters
```python
# In demo_load_test.py
num_requests = 100        # Total requests
concurrent_requests = 20  # Concurrent connections
```

### Add New Test Scenarios
```python
# In locustfile.py
@task(weight)
def new_test_scenario(self):
    # Custom test logic
```

### Container Resource Limits
```yaml
# In docker-compose.yml
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
```

## 🔍 Monitoring & Analysis

The suite provides comprehensive metrics:
- **Throughput**: Requests per second
- **Latency**: Response time percentiles
- **Reliability**: Success/failure rates
- **Scalability**: Performance vs. container count
- **Resource Usage**: Memory and CPU utilization

## 🎉 Benefits

1. **Performance Validation**: Verify model can handle production load
2. **Scaling Insights**: Understand horizontal scaling benefits
3. **Bottleneck Identification**: Find performance limitations
4. **Capacity Planning**: Determine optimal container configuration
5. **Regression Testing**: Monitor performance over time

This load testing suite provides everything needed to validate and optimize the weather prediction model's performance under various load conditions and container configurations.