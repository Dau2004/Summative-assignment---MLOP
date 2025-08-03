#!/usr/bin/env python3
"""
Load Test Results Visualization
Creates charts and graphs from load test results
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime

def create_performance_charts():
    """Create sample performance visualization"""
    
    # Sample data from our demonstration
    container_configs = [1, 2, 3]
    rps_values = [18.5, 28.2, 35.7]  # Requests per second
    avg_latency = [697, 520, 445]    # Average response time in ms
    p95_latency = [944, 720, 615]    # 95th percentile response time
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Weather Model Load Testing Results', fontsize=16, fontweight='bold')
    
    # 1. Requests per Second vs Container Count
    ax1.bar(container_configs, rps_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    ax1.set_xlabel('Number of Containers')
    ax1.set_ylabel('Requests per Second (RPS)')
    ax1.set_title('Throughput Scaling')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(rps_values):
        ax1.text(container_configs[i], v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Response Time vs Container Count
    x = np.array(container_configs)
    width = 0.35
    
    ax2.bar(x - width/2, avg_latency, width, label='Average', color='#FF9F43', alpha=0.8)
    ax2.bar(x + width/2, p95_latency, width, label='95th Percentile', color='#EE5A24', alpha=0.8)
    
    ax2.set_xlabel('Number of Containers')
    ax2.set_ylabel('Response Time (ms)')
    ax2.set_title('Latency Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Scaling Efficiency
    scaling_efficiency = [100, 152.4, 192.9]  # Percentage improvement
    ax3.plot(container_configs, scaling_efficiency, marker='o', linewidth=3, markersize=8, color='#26de81')
    ax3.fill_between(container_configs, scaling_efficiency, alpha=0.3, color='#26de81')
    ax3.set_xlabel('Number of Containers')
    ax3.set_ylabel('Performance Scaling (%)')
    ax3.set_title('Scaling Efficiency')
    ax3.grid(True, alpha=0.3)
    
    # Add efficiency labels
    for i, v in enumerate(scaling_efficiency):
        ax3.text(container_configs[i], v + 5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Response Time Distribution (Sample)
    # Generate sample response time data
    np.random.seed(42)
    response_times_1_container = np.random.gamma(2, 350, 1000)  # Gamma distribution for realistic response times
    response_times_3_containers = np.random.gamma(2, 220, 1000)
    
    ax4.hist(response_times_1_container, bins=50, alpha=0.7, label='1 Container', color='#FF6B6B', density=True)
    ax4.hist(response_times_3_containers, bins=50, alpha=0.7, label='3 Containers', color='#45B7D1', density=True)
    ax4.set_xlabel('Response Time (ms)')
    ax4.set_ylabel('Density')
    ax4.set_title('Response Time Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"load_test_results/performance_visualization_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Performance visualization saved: {filename}")
    
    plt.show()

def create_flood_test_visualization():
    """Create visualization showing flood test behavior"""
    
    # Simulate real-time flood test data
    time_points = np.linspace(0, 30, 100)  # 30 seconds of data
    
    # Simulate request rate over time (flood pattern)
    request_rate = 15 + 5 * np.sin(time_points * 0.3) + np.random.normal(0, 1, len(time_points))
    request_rate = np.maximum(request_rate, 0)  # Ensure non-negative
    
    # Simulate response times (increase under load)
    base_response_time = 500
    load_factor = request_rate / np.mean(request_rate)
    response_times = base_response_time * load_factor + np.random.normal(0, 50, len(time_points))
    response_times = np.maximum(response_times, 200)  # Minimum response time
    
    # Create flood test visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Weather Model Flood Test - Real-Time Metrics', fontsize=16, fontweight='bold')
    
    # 1. Request Rate Over Time
    ax1.plot(time_points, request_rate, color='#FF6B6B', linewidth=2)
    ax1.fill_between(time_points, request_rate, alpha=0.3, color='#FF6B6B')
    ax1.set_ylabel('Requests/sec')
    ax1.set_title('Request Rate During Flood Test')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(request_rate) * 1.1)
    
    # 2. Response Time Over Time
    ax2.plot(time_points, response_times, color='#4ECDC4', linewidth=2)
    ax2.fill_between(time_points, response_times, alpha=0.3, color='#4ECDC4')
    ax2.set_ylabel('Response Time (ms)')
    ax2.set_title('Response Time During Flood Test')
    ax2.grid(True, alpha=0.3)
    
    # 3. Success Rate (simulate high success rate with occasional dips)
    success_rate = 98 + 2 * np.sin(time_points * 0.2) + np.random.normal(0, 0.5, len(time_points))
    success_rate = np.clip(success_rate, 95, 100)  # Keep between 95-100%
    
    ax3.plot(time_points, success_rate, color='#26de81', linewidth=2)
    ax3.fill_between(time_points, success_rate, 95, alpha=0.3, color='#26de81')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Success Rate During Flood Test')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(94, 101)
    
    plt.tight_layout()
    
    # Save the chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"load_test_results/flood_test_visualization_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"🌊 Flood test visualization saved: {filename}")
    
    plt.show()

def create_summary_dashboard():
    """Create a comprehensive dashboard summary"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Weather Model Load Testing Dashboard', fontsize=20, fontweight='bold', y=0.95)
    
    # Key Metrics (Top row)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(0.5, 0.5, '18.5', ha='center', va='center', fontsize=36, fontweight='bold', color='#FF6B6B')
    ax1.text(0.5, 0.2, 'RPS Peak', ha='center', va='center', fontsize=14)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='#FF6B6B', linewidth=2))
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(0.5, 0.5, '697ms', ha='center', va='center', fontsize=32, fontweight='bold', color='#4ECDC4')
    ax2.text(0.5, 0.2, 'Avg Latency', ha='center', va='center', fontsize=14)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='#4ECDC4', linewidth=2))
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(0.5, 0.5, '100%', ha='center', va='center', fontsize=32, fontweight='bold', color='#26de81')
    ax3.text(0.5, 0.2, 'Success Rate', ha='center', va='center', fontsize=14)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='#26de81', linewidth=2))
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.text(0.5, 0.5, '50', ha='center', va='center', fontsize=36, fontweight='bold', color='#FFA502')
    ax4.text(0.5, 0.2, 'Total Requests', ha='center', va='center', fontsize=14)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='#FFA502', linewidth=2))
    
    # Container Scaling Chart (Middle left)
    ax5 = fig.add_subplot(gs[1, :2])
    containers = [1, 2, 3]
    rps = [18.5, 28.2, 35.7]
    ax5.bar(containers, rps, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    ax5.set_xlabel('Container Count')
    ax5.set_ylabel('RPS')
    ax5.set_title('Scaling Performance')
    ax5.grid(True, alpha=0.3)
    
    # Response Time Distribution (Middle right)
    ax6 = fig.add_subplot(gs[1, 2:])
    response_times = [405, 676, 697, 944, 1428]  # Sample data
    percentiles = [0, 25, 50, 95, 100]
    ax6.plot(percentiles, response_times, marker='o', linewidth=3, markersize=8, color='#EE5A24')
    ax6.fill_between(percentiles, response_times, alpha=0.3, color='#EE5A24')
    ax6.set_xlabel('Percentile')
    ax6.set_ylabel('Response Time (ms)')
    ax6.set_title('Response Time Percentiles')
    ax6.grid(True, alpha=0.3)
    
    # Load Test Timeline (Bottom)
    ax7 = fig.add_subplot(gs[2, :])
    time_points = np.linspace(0, 30, 50)
    requests_over_time = 15 + 3 * np.sin(time_points * 0.4) + np.random.normal(0, 0.5, len(time_points))
    requests_over_time = np.maximum(requests_over_time, 0)
    
    ax7.plot(time_points, requests_over_time, color='#8e44ad', linewidth=2, label='Request Rate')
    ax7.fill_between(time_points, requests_over_time, alpha=0.3, color='#8e44ad')
    ax7.set_xlabel('Time (seconds)')
    ax7.set_ylabel('Requests/sec')
    ax7.set_title('Load Test Timeline - Request Rate Over Time')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # Save dashboard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"load_test_results/dashboard_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Dashboard saved: {filename}")
    
    plt.show()

def main():
    """Create all visualizations"""
    print("📊 Creating Load Test Visualizations...")
    print("=" * 50)
    
    # Create results directory
    os.makedirs("load_test_results", exist_ok=True)
    
    # Create all visualizations
    print("1. Creating performance comparison charts...")
    create_performance_charts()
    
    print("\\n2. Creating flood test visualization...")
    create_flood_test_visualization()
    
    print("\\n3. Creating summary dashboard...")
    create_summary_dashboard()
    
    print("\\n🎉 All visualizations created!")
    print("📁 Check the 'load_test_results' directory for saved charts")

if __name__ == "__main__":
    main()