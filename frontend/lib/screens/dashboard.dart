import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/api_service.dart';
import '../widgets/class_distribution_chart.dart';
import '../widgets/training_performance_chart.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({Key? key}) : super(key: key);

  static final GlobalKey<_DashboardScreenState> globalKey = GlobalKey<_DashboardScreenState>();

  static void refreshDashboard() {
    globalKey.currentState?._refreshAllData();
  }

  static void refreshAfterTraining() {
    globalKey.currentState?._refreshAfterTraining();
  }

  @override
  _DashboardScreenState createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> with WidgetsBindingObserver {
  late Timer _timer;
  
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    
    // Auto-fetch data when dashboard loads
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _refreshAllData();
    });
    
    // Set up a timer to update the time display every 10 seconds for real-time updates
    _timer = Timer.periodic(const Duration(seconds: 10), (timer) {
      if (mounted) {
        setState(() {
          // This will trigger a rebuild to update the time display
        });
      }
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _timer.cancel();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    super.didChangeAppLifecycleState(state);
    if (state == AppLifecycleState.resumed) {
      // Refresh data when app comes back to foreground
      _refreshAllData();
    }
  }

  Future<void> _refreshAllData() async {
    final apiService = Provider.of<ApiService>(context, listen: false);
    await apiService.fetchModelStatus();
    await apiService.fetchTrainingPerformance();
    
    // Force UI update after data refresh
    if (mounted) {
      setState(() {});
    }
  }

  Future<void> _refreshAfterTraining() async {
    // Show immediate feedback that training completed
    if (mounted) {
      setState(() {});
    }
    
    // Wait a moment then refresh data
    await Future.delayed(const Duration(milliseconds: 500));
    await _refreshAllData();
    
    // Show success message
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Model retrained successfully! Dashboard updated.'),
          backgroundColor: Colors.green,
          duration: Duration(seconds: 3),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final apiService = Provider.of<ApiService>(context);

    return Scaffold(
      backgroundColor: Theme.of(context).colorScheme.background,
      appBar: AppBar(
        title: const Text('Weather Dashboard', style: TextStyle(fontWeight: FontWeight.bold)),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => _refreshAllData(),
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: () => _refreshAllData(),
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            children: [
              // Model Status Card - Enhanced
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          const Icon(Icons.model_training, color: Colors.blue),
                          const SizedBox(width: 8),
                          const Expanded(
                            child: Text(
                              'Model Training Status',
                              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                            ),
                          ),
                          const Spacer(),
                          if (apiService.isLoading)
                            const SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            ),
                        ],
                      ),
                      const SizedBox(height: 16),
                      _buildStatusItem('Model Version', apiService.modelStatus?['version'] ?? 'N/A'),
                      _buildStatusItem('Last Trained', _formatTrainingDate(apiService.modelStatus?['last_trained'])),
                      if (apiService.modelStatus?['training_time'] != null)
                        _buildStatusItem('Training Time', apiService.modelStatus!['training_time']),
                      // if (apiService.modelStatus?['training_type'] != null)
                      //   _buildStatusItem('Training Type', apiService.modelStatus!['training_type']),
                      _buildStatusItem('Training Status', _getTrainingStatus(apiService.modelStatus?['status'])),
                      if (apiService.modelStatus?['total_images'] != null)
                        _buildStatusItem('Training Dataset Size', '${apiService.modelStatus!['total_images']} images'),
                      // if (apiService.modelStatus?['improvement_note'] != null && apiService.modelStatus!['improvement_note'].toString().isNotEmpty)
                      //   _buildStatusItem('Latest Update', apiService.modelStatus!['improvement_note']),
                      const SizedBox(height: 12),
                      _buildTrainingInterpretation(apiService.modelStatus),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 16),

              // Performance Metrics Card - Enhanced with Graphs
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          const Icon(Icons.analytics, color: Colors.green),
                          const SizedBox(width: 8),
                          const Expanded(
                            child: Text(
                              'Model Performance After Training',
                              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
                      _buildPerformanceChart(apiService.modelStatus),
                      const SizedBox(height: 16),
                      _buildMetricItem(
                        'Accuracy',
                        apiService.modelStatus?['accuracy']?.toString() ?? 'N/A',
                        1.0,
                      ),
                      const SizedBox(height: 8),
                      _buildMetricItem(
                        'Precision',
                        apiService.modelStatus?['precision']?.toString() ?? 'N/A',
                        1.0,
                      ),
                      const SizedBox(height: 8),
                      _buildMetricItem(
                        'Recall',
                        apiService.modelStatus?['recall']?.toString() ?? 'N/A',
                        1.0,
                      ),
                      const SizedBox(height: 12),
                      _buildPerformanceInterpretation(apiService.modelStatus),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 16),

              // Training Performance Graph Card - New Addition
              if (apiService.trainingPerformance != null)
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            const Icon(Icons.trending_up, color: Colors.purple),
                            const SizedBox(width: 8),
                            const Expanded(
                              child: Text(
                                'Training Performance Over Epochs',
                                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),
                        TrainingPerformanceChart(
                          trainingData: apiService.trainingPerformance,
                        ),
                        const SizedBox(height: 12),
                        _buildTrainingProgressInterpretation(apiService.trainingPerformance),
                      ],
                    ),
                  ),
                ),

              if (apiService.trainingPerformance != null)
                const SizedBox(height: 16),

              // Class Distribution Card - Enhanced with Interpretation
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          const Icon(Icons.pie_chart, color: Colors.orange),
                          const SizedBox(width: 8),
                          const Expanded(
                            child: Text(
                              'Training Data Distribution',
                              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
                      SizedBox(
                        height: 200,
                        child: ClassDistributionChart(
                          data: apiService.modelStatus?['class_distribution'] ?? {},
                        ),
                      ),
                      const SizedBox(height: 12),
                      _buildClassDistributionDetails(apiService.modelStatus),
                      const SizedBox(height: 8),
                      _buildDataBalanceInterpretation(apiService.modelStatus),
                    ],
                  ),
                ),
              ),

              // UPDATED: Better message display
              if (apiService.message.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.all(16),
                  child: Text(
                    apiService.message,
                    style: TextStyle(
                      color: apiService.message.contains('Error')
                          ? Colors.red
                          : Colors.green,
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStatusItem(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '$label: ',
            style: const TextStyle(fontWeight: FontWeight.bold),
          ),
          Expanded(
            child: Text(
              value,
              overflow: TextOverflow.ellipsis,
              maxLines: 2,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMetricItem(String label, String value, double target) {
    final numericValue = double.tryParse(value) ?? 0;
    final progress = numericValue / target;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Expanded(
              child: Text(
                label,
                style: const TextStyle(fontWeight: FontWeight.bold),
                overflow: TextOverflow.ellipsis,
              ),
            ),
            Text('${(numericValue * 100).toStringAsFixed(1)}%'),
          ],
        ),
        const SizedBox(height: 4),
        LinearProgressIndicator(
          value: progress > 1.0 ? 1.0 : progress,
          backgroundColor: Colors.grey[300],
          color: progress >= 0.9
              ? Colors.green
              : progress >= 0.7
                  ? Colors.orange
                  : Colors.red,
        ),
      ],
    );
  }

  String _formatTrainingDate(String? dateStr) {
    if (dateStr == null) return 'Never trained';
    try {
      final date = DateTime.parse(dateStr);
      final now = DateTime.now();
      final difference = now.difference(date);
      
      if (difference.inSeconds < 10) return 'Just now ⚡';
      if (difference.inSeconds < 60) return '${difference.inSeconds} seconds ago 🔥';
      if (difference.inMinutes < 2) return '1 minute ago 🟢';
      if (difference.inMinutes < 60) return '${difference.inMinutes} minutes ago 🟢';
      if (difference.inHours < 2) return '1 hour ago';
      if (difference.inHours < 24) return '${difference.inHours} hours ago';
      if (difference.inDays == 1) return 'Yesterday';
      if (difference.inDays < 7) return '${difference.inDays} days ago';
      
      // Format as readable date for older dates
      return '${date.day}/${date.month}/${date.year}';
    } catch (e) {
      return dateStr;
    }
  }

  String _getTrainingStatus(String? status) {
    switch (status?.toLowerCase()) {
      case 'active':
        return '🟢 Active & Ready';
      case 'training':
        return '🟡 Training in Progress';
      case 'error':
        return '🔴 Training Error';
      default:
        return '⚪ Unknown Status';
    }
  }

  Widget _buildTrainingInterpretation(Map<String, dynamic>? modelStatus) {
    final totalImages = modelStatus?['total_images'] ?? 0;
    
    String interpretation;
    Color bgColor;
    
    if (totalImages < 50) {
      interpretation = 'Small dataset: Model may overfit. Consider adding more training data for better generalization.';
      bgColor = Colors.orange.shade50;
    } else if (totalImages < 200) {
      interpretation = 'Moderate dataset: Good for initial training. More data could improve performance.';
      bgColor = Colors.blue.shade50;
    } else {
      interpretation = 'Large dataset: Excellent for robust model training and generalization.';
      bgColor = Colors.green.shade50;
    }

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.grey.shade300),
      ),
      child: Row(
        children: [
          const Icon(Icons.lightbulb, size: 20, color: Colors.amber),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              interpretation,
              style: const TextStyle(fontSize: 12, fontStyle: FontStyle.italic),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPerformanceChart(Map<String, dynamic>? modelStatus) {
    final accuracy = (modelStatus?['accuracy'] ?? 0.0) * 100;
    final precision = (modelStatus?['precision'] ?? 0.0) * 100;
    final recall = (modelStatus?['recall'] ?? 0.0) * 100;

    return Container(
      height: 140,
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Expanded(child: _buildCircularMetric('Accuracy', accuracy, Colors.blue)),
          Expanded(child: _buildCircularMetric('Precision', precision, Colors.green)),
          Expanded(child: _buildCircularMetric('Recall', recall, Colors.orange)),
        ],
      ),
    );
  }

  Widget _buildCircularMetric(String label, double value, Color color) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        SizedBox(
          width: 60,
          height: 60,
          child: Stack(
            alignment: Alignment.center,
            children: [
              SizedBox(
                width: 60,
                height: 60,
                child: CircularProgressIndicator(
                  value: value / 100,
                  strokeWidth: 6,
                  backgroundColor: Colors.grey.shade300,
                  valueColor: AlwaysStoppedAnimation<Color>(color),
                ),
              ),
              Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  color: Colors.white,
                  shape: BoxShape.circle,
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.1),
                      blurRadius: 2,
                      offset: const Offset(0, 1),
                    ),
                  ],
                ),
                child: Center(
                  child: Text(
                    '${value.toStringAsFixed(0)}%',
                    style: const TextStyle(
                      fontSize: 10, 
                      fontWeight: FontWeight.bold,
                      color: Colors.black87,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 8),
        Text(
          label,
          style: const TextStyle(
            fontSize: 10,
            fontWeight: FontWeight.w500,
          ),
          textAlign: TextAlign.center,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
        ),
      ],
    );
  }

  Widget _buildPerformanceInterpretation(Map<String, dynamic>? modelStatus) {
    final accuracy = modelStatus?['accuracy'] ?? 0.0;
    final precision = modelStatus?['precision'] ?? 0.0;
    final recall = modelStatus?['recall'] ?? 0.0;
    
    String interpretation;
    Color bgColor;
    
    if (accuracy >= 0.9 && precision >= 0.85 && recall >= 0.85) {
      interpretation = 'Excellent model performance! High accuracy with balanced precision and recall.';
      bgColor = Colors.green.shade50;
    } else if (accuracy >= 0.8) {
      interpretation = 'Good model performance. Consider fine-tuning or adding more training data.';
      bgColor = Colors.blue.shade50;
    } else {
      interpretation = 'Model needs improvement. Consider retraining with more diverse data.';
      bgColor = Colors.orange.shade50;
    }

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.grey.shade300),
      ),
      child: Row(
        children: [
          const Icon(Icons.psychology, size: 20, color: Colors.purple),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              interpretation,
              style: const TextStyle(fontSize: 12, fontStyle: FontStyle.italic),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildClassDistributionDetails(Map<String, dynamic>? modelStatus) {
    final distribution = modelStatus?['class_distribution'] as Map<String, dynamic>? ?? {};
    final total = distribution.values.fold<int>(0, (sum, count) => sum + (count as int? ?? 0));
    
    return Column(
      children: distribution.entries.map((entry) {
        final percentage = total > 0 ? (entry.value / total * 100) : 0.0;
        return Padding(
          padding: const EdgeInsets.symmetric(vertical: 2),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                flex: 2,
                child: Text('${entry.key}:', style: const TextStyle(fontWeight: FontWeight.w500)),
              ),
              Expanded(
                flex: 3,
                child: Text(
                  '${entry.value} images (${percentage.toStringAsFixed(1)}%)',
                  textAlign: TextAlign.right,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          ),
        );
      }).toList(),
    );
  }

  Widget _buildDataBalanceInterpretation(Map<String, dynamic>? modelStatus) {
    final distribution = modelStatus?['class_distribution'] as Map<String, dynamic>? ?? {};
    final counts = distribution.values.map((v) => v as int? ?? 0).toList();
    
    if (counts.isEmpty) {
      return const SizedBox.shrink();
    }
    
    final min = counts.reduce((a, b) => a < b ? a : b);
    final max = counts.reduce((a, b) => a > b ? a : b);
    final imbalanceRatio = max / (min > 0 ? min : 1);
    
    String interpretation;
    Color bgColor;
    
    if (imbalanceRatio <= 1.5) {
      interpretation = 'Well-balanced dataset! Classes have similar representation.';
      bgColor = Colors.green.shade50;
    } else if (imbalanceRatio <= 2.5) {
      interpretation = 'Moderate class imbalance. Model should perform reasonably well.';
      bgColor = Colors.blue.shade50;
    } else {
      interpretation = 'High class imbalance detected. Consider data augmentation for minority classes.';
      bgColor = Colors.orange.shade50;
    }

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.grey.shade300),
      ),
      child: Row(
        children: [
          const Icon(Icons.balance, size: 20, color: Colors.teal),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              interpretation,
              style: const TextStyle(fontSize: 12, fontStyle: FontStyle.italic),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTrainingProgressInterpretation(Map<String, dynamic>? trainingData) {
    if (trainingData == null) return const SizedBox.shrink();
    
    final accuracy = trainingData['accuracy'] ?? 0.0;
    final valAccuracy = trainingData['val_accuracy'] ?? 0.0;
    final epochs = trainingData['epochs_completed'] ?? 1;
    
    String interpretation;
    Color bgColor;
    
    final accuracyGap = (accuracy - valAccuracy).abs();
    
    if (accuracyGap > 0.1) {
      interpretation = 'Overfitting detected! Training accuracy significantly exceeds validation accuracy. Consider more training data or regularization.';
      bgColor = Colors.red.shade50;
    } else if (accuracy < 0.7) {
      interpretation = 'Model needs more training. Consider increasing epochs or improving data quality.';
      bgColor = Colors.orange.shade50;
    } else if (epochs < 5) {
      interpretation = 'Short training session completed. Model converged quickly, possibly due to small dataset.';
      bgColor = Colors.blue.shade50;
    } else {
      interpretation = 'Good training progression! Model shows balanced learning on training and validation data.';
      bgColor = Colors.green.shade50;
    }

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.grey.shade300),
      ),
      child: Row(
        children: [
          const Icon(Icons.insights, size: 20, color: Colors.indigo),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              interpretation,
              style: const TextStyle(fontSize: 12, fontStyle: FontStyle.italic),
            ),
          ),
        ],
      ),
    );
  }
}