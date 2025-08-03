#!/bin/bash

echo "🔥 Weather Model Load Testing Suite"
echo "=================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if required Python packages are installed
echo "📦 Checking dependencies..."
python3 -c "import locust, aiohttp, matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Installing required packages..."
    pip3 install -r requirements.txt
fi

# Create results directory
mkdir -p load_test_results

echo ""
echo "Choose a load testing option:"
echo "1. Quick Demo (100 requests, real-time visualization)"
echo "2. Comprehensive Test (Multiple container configurations)"
echo "3. Single Container Test (Locust Web UI)"
echo "4. Build Docker Images Only"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "🚀 Running Quick Load Test Demo..."
        echo "This will send 100 requests with real-time metrics display"
        echo ""
        
        # Start single container for quick test
        echo "Starting single container..."
        docker-compose up -d weather-api-1
        
        # Wait for container to be ready
        echo "Waiting for container to be ready..."
        sleep 10
        
        # Check if container is healthy
        for i in {1..12}; do
            if curl -s http://localhost:8001/status > /dev/null; then
                echo "✅ Container is ready!"
                break
            fi
            echo "⏳ Waiting for container... ($i/12)"
            sleep 5
        done
        
        # Run quick test
        python3 quick_load_test.py
        
        # Cleanup
        docker-compose down
        ;;
        
    2)
        echo "🎯 Running Comprehensive Load Test Suite..."
        echo "This will test 1, 2, and 3 container configurations"
        echo "⚠️  This may take 15-20 minutes to complete"
        echo ""
        
        read -p "Continue? (y/N): " confirm
        if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
            python3 load_test_runner.py
        else
            echo "Test cancelled."
        fi
        ;;
        
    3)
        echo "🌐 Starting Locust Web UI..."
        echo "This will start a single container and Locust web interface"
        echo ""
        
        # Start single container
        echo "Starting container..."
        docker-compose up -d weather-api-1
        
        # Wait for container
        echo "Waiting for container to be ready..."
        sleep 10
        
        # Start Locust web UI
        echo "Starting Locust Web UI at http://localhost:8089"
        echo "Target host: http://localhost:8001"
        echo "Press Ctrl+C to stop"
        
        locust -f locustfile.py --host http://localhost:8001
        
        # Cleanup
        docker-compose down
        ;;
        
    4)
        echo "🏗️  Building Docker Images..."
        docker-compose build
        echo "✅ Docker images built successfully!"
        ;;
        
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "🎉 Load testing completed!"
echo "📊 Check the 'load_test_results' directory for detailed reports"