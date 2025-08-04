import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import '../services/api_service.dart';

class PredictScreen extends StatefulWidget {
  const PredictScreen({super.key});

  @override
  State<PredictScreen> createState() => _PredictScreenState();
}

class _PredictScreenState extends State<PredictScreen> {
  File? _selectedImage;
  Map<String, dynamic>? _predictionResult;

  Future<void> _pickImage() async {
    final pickedFile = await ImagePicker().pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _selectedImage = File(pickedFile.path);
        _predictionResult = null;
      });
    }
  }

  Future<void> _predictWeather() async {
    if (_selectedImage == null) return;

    try {
      final apiService = Provider.of<ApiService>(context, listen: false);
      final result = await apiService.predictWeather(_selectedImage!);
      
      setState(() => _predictionResult = result);
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e')),
      );
    }
  }

  String _getConfidenceText() {
    if (_predictionResult == null) return 'N/A';
    
    final confidence = _predictionResult!['confidence'];
    if (confidence == null) return 'N/A';
    
    try {
      final percentage = (confidence * 100).toStringAsFixed(1);
      return '$percentage%';
    } catch (e) {
      return 'N/A';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).colorScheme.background,
      appBar: AppBar(
        title: const Text('Weather Prediction', style: TextStyle(fontWeight: FontWeight.bold)),
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
                      'Upload a weather image',
                      style: TextStyle(fontSize: 18),
                    ),
                    const SizedBox(height: 16),
                    ElevatedButton(
                      onPressed: _pickImage,
                      child: const Text('Select Image'),
                    ),
                    if (_selectedImage != null) ...[
                      const SizedBox(height: 16),
                      SizedBox(
                        height: 200,
                        child: Image.file(_selectedImage!, fit: BoxFit.cover),
                      ),
                      const SizedBox(height: 16),
                      FilledButton(
                        onPressed: _predictWeather,
                        child: const Text('Predict Weather'),
                      ),
                    ],
                  ],
                ),
              ),
            ),
            if (_predictionResult != null) ...[
              const SizedBox(height: 16),
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Prediction Result',
                        style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Weather: ${_predictionResult!['prediction'] ?? 'Unknown'}',  // Handle null prediction
                        style: const TextStyle(fontSize: 16),
                      ),
                      Text(
                        'Confidence: ${_getConfidenceText()}',  // Use helper method
                        style: const TextStyle(fontSize: 16),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}