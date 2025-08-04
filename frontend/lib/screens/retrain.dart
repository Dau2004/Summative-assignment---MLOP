import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:weather_classifier/services/api_service.dart';
import 'package:weather_classifier/screens/dashboard.dart';

class RetrainScreen extends StatefulWidget {
  const RetrainScreen({super.key});

  @override
  State<RetrainScreen> createState() => _RetrainScreenState();
}

class _RetrainScreenState extends State<RetrainScreen> {
  @override
  Widget build(BuildContext context) {
    final apiService = Provider.of<ApiService>(context);

    return Scaffold(
      backgroundColor: Theme.of(context).colorScheme.background,
      appBar: AppBar(
        title: const Text('Model Training', style: TextStyle(fontWeight: FontWeight.bold)),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  children: [
                    const Text(
                      'Retrain the model with new data',
                      style: TextStyle(fontSize: 18),
                    ),
                    const SizedBox(height: 16),
                    const Text(
                      'After uploading new images, you can retrain the model to improve its accuracy',
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: apiService.isLoading ? null : () => _retrainModel(context),
              child: apiService.isLoading
                  ? const CircularProgressIndicator()
                  : const Text('Start Retraining'),
            ),
            if (apiService.message.isNotEmpty)
              Padding(
                padding: const EdgeInsets.all(16),
                child: Text(apiService.message),
              ),
          ],
        ),
      ),
    );
  }

  Future<void> _retrainModel(BuildContext context) async {
    try {
      final apiService = Provider.of<ApiService>(context, listen: false);
      
      // Show confirmation dialog
      final confirmed = await showDialog<bool>(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('Confirm Retraining'),
          content: const Text('Are you sure you want to retrain the model? This process may take several minutes.'),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(false),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () => Navigator.of(context).pop(true),
              child: const Text('Start'),
            ),
          ],
        ),
      );
      
      if (confirmed != true) return;
      
      final result = await apiService.triggerRetraining();
      await apiService.fetchModelStatus();
      
      // Refresh dashboard to show updated training status immediately
      DashboardScreen.refreshAfterTraining();
      
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(result['message'] ?? 'Model retrained successfully'),
            backgroundColor: Colors.green,
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }
}