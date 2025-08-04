import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:file_picker/file_picker.dart';

class ApiService with ChangeNotifier {
  final String baseUrl;
  bool _isLoading = false;
  String _message = '';
  Map<String, dynamic>? _modelStatus;
  Map<String, dynamic>? _trainingPerformance;
  Map<String, dynamic>? _databaseStats;
  List<dynamic>? _databaseImages;

  ApiService({required this.baseUrl});

  bool get isLoading => _isLoading;
  String get message => _message;
  Map<String, dynamic>? get modelStatus => _modelStatus;
  Map<String, dynamic>? get trainingPerformance => _trainingPerformance;
  Map<String, dynamic>? get databaseStats => _databaseStats;
  List<dynamic>? get databaseImages => _databaseImages;

  Future<Map<String, dynamic>> predictWeather(File imageFile) async {
    _setLoading(true, 'Predicting weather...');
    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/predict'),
      );
      request.files.add(
        await http.MultipartFile.fromPath(
          'image',
          imageFile.path,
        ),
      );

      var response = await request.send();
      final respStr = await response.stream.bytesToString();
      _setLoading(false, 'Prediction completed');
      return json.decode(respStr);
    } catch (e) {
      _setLoading(false, 'Error: $e');
      rethrow;
    }
  }

  Future<Map<String, dynamic>> uploadBulkFiles(List<PlatformFile> files) async {
    _setLoading(true, 'Uploading files...');
    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/upload'),
      );

      // Add files with correct field name
      for (var file in files) {
        if (file.path != null) {
          request.files.add(
            await http.MultipartFile.fromPath(
              'files', // Change from 'images' to 'files' to match backend
              file.path!,
              filename: file.name,
            ),
          );
        }
      }

      var response = await request.send();
      final respStr = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        _setLoading(false, 'Upload completed successfully');
        return json.decode(respStr);
      } else {
        final error = json.decode(respStr);
        throw Exception(error['detail'] ?? 'Upload failed');
      }
    } catch (e) {
      _setLoading(false, 'Upload failed: $e');
      rethrow;
    }
  }

  Future<Map<String, dynamic>> triggerRetraining({int epochs = 10}) async {
    _setLoading(true, 'Starting retraining...');
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/retrain?epochs=$epochs'),
        headers: {'Content-Type': 'application/json'},
      );
      
      final responseData = json.decode(response.body);
      
      if (response.statusCode == 200) {
        _setMessage('Retraining completed successfully');
        
        // Update model status with current timestamp
        if (_modelStatus != null) {
          _modelStatus!['last_trained'] = DateTime.now().toIso8601String();
          _modelStatus!['training_date'] = DateTime.now().toIso8601String();
        }
        
        // Auto-refresh model status after successful retraining
        await fetchModelStatus();
        
        // Force refresh training performance to get latest data
        await fetchTrainingPerformance();
        
        // Store training performance data from retraining response
        if (responseData.containsKey('accuracy')) {
          _trainingPerformance = {
            'accuracy': responseData['accuracy'],
            'val_accuracy': responseData['val_accuracy'] ?? responseData['accuracy'],
            'loss': responseData['loss'] ?? 0.0,
            'val_loss': responseData['val_loss'] ?? 0.0,
            'epochs_completed': responseData['epochs_completed'] ?? epochs,
            'training_type': responseData['training_type'] ?? 'Training',
            'improvement_note': responseData['improvement_note'] ?? '',
          };
        }
        
        _setLoading(false, 'Model retrained successfully - Dashboard updated');
        return responseData;
      } else if (response.statusCode == 400) {
        final errorMessage = responseData['detail'] ?? 'Bad request';
        _setMessage('Error: $errorMessage');
        throw Exception(errorMessage);
      } else {
        final errorMessage = responseData['detail'] ?? 'Failed to start retraining';
        _setMessage('Error: $errorMessage');
        throw Exception(errorMessage);
      }
    } catch (e) {
      if (!_message.startsWith('Error:')) {
        _setMessage('Error: $e');
      }
      rethrow;
    } finally {
      _setLoading(false, 'Retraining process finished');
    }
  }

  Future<void> fetchModelStatus() async {
    _setLoading(true, 'Fetching model status...');
    try {
      // Change from /status to /model/status
      final response = await http.get(Uri.parse('$baseUrl/model/status'));
      
      if (response.statusCode == 200) {
        _modelStatus = json.decode(response.body);
        
        // Check if this response contains training performance data
        if (_modelStatus?.containsKey('accuracy') == true) {
          _trainingPerformance = {
            'accuracy': _modelStatus!['accuracy'],
            'val_accuracy': _modelStatus!['val_accuracy'] ?? _modelStatus!['accuracy'],
            'loss': _modelStatus!['loss'] ?? 0.0,
            'val_loss': _modelStatus!['val_loss'] ?? 0.0,
            'epochs_completed': _modelStatus!['epochs_completed'] ?? 1,
          };
        }
        
        // Fetch detailed training performance data
        await fetchTrainingPerformance();
        
        _setLoading(false, 'Status updated');
      } else {
        throw Exception('Failed to fetch status: ${response.statusCode}');
      }
      notifyListeners();
    } catch (e) {
      _setLoading(false, 'Failed to fetch status: $e');
      print('Error fetching status: $e'); // Debug log
      rethrow;
    }
  }

  // Add method to fetch detailed training performance
  Future<void> fetchTrainingPerformance() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/training/performance'));
      if (response.statusCode == 200) {
        final performanceData = json.decode(response.body);
        if (performanceData.containsKey('accuracy_history')) {
          _trainingPerformance = performanceData;
        }
      }
    } catch (e) {
      print('Error fetching training performance: $e');
    }
  }

  // New database-related methods
  Future<void> fetchDatabaseStats() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/database/stats'));
      if (response.statusCode == 200) {
        _databaseStats = json.decode(response.body);
        notifyListeners();
      }
    } catch (e) {
      print('Error fetching database stats: $e');
    }
  }

  Future<void> fetchDatabaseImages() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/database/images'));
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        _databaseImages = data['images'];
        notifyListeners();
      }
    } catch (e) {
      print('Error fetching database images: $e');
    }
  }

  Future<Map<String, dynamic>> deleteImage(int imageId) async {
    try {
      final response = await http.delete(Uri.parse('$baseUrl/database/images/$imageId'));
      if (response.statusCode == 200) {
        // Refresh the database images after deletion
        await fetchDatabaseImages();
        await fetchDatabaseStats();
        return json.decode(response.body);
      } else {
        throw Exception('Failed to delete image');
      }
    } catch (e) {
      print('Error deleting image: $e');
      rethrow;
    }
  }

  Future<Map<String, dynamic>> cleanupDatabase() async {
    _setLoading(true, 'Cleaning up database...');
    try {
      final response = await http.post(Uri.parse('$baseUrl/database/cleanup'));
      final result = json.decode(response.body);
      
      // Refresh data after cleanup
      await fetchDatabaseStats();
      await fetchDatabaseImages();
      
      _setLoading(false, 'Database cleanup completed');
      return result;
    } catch (e) {
      _setLoading(false, 'Cleanup failed: $e');
      rethrow;
    }
  }

  void _setLoading(bool loading, String message) {
    _isLoading = loading;
    _message = message;
    notifyListeners();
  }

  void _setMessage(String message) {
    _message = message;
    notifyListeners();
  }
}