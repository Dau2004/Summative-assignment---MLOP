import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import 'package:weather_classifier/services/api_service.dart';
import 'dart:io';

class UploadScreen extends StatefulWidget {
  const UploadScreen({super.key});

  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  List<PlatformFile> _selectedFiles = [];
  final ImagePicker _imagePicker = ImagePicker();

  // For iOS Simulator - use file picker to access laptop files
  Future<void> _pickFilesFromComputer() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowMultiple: true,
      allowedExtensions: ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'],
    );

    if (result != null) {
      // Optional: Additional validation
      final validFiles = result.files.where((file) {
        final extension = file.extension?.toLowerCase();
        return extension == 'jpg' || extension == 'jpeg' || extension == 'png';
      }).toList();
      
      if (validFiles.isEmpty) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Please select valid image files (jpg, jpeg, png)')),
        );
        return;
      }
      
      setState(() => _selectedFiles = validFiles);
    }
  }

  // For physical devices - use image picker for camera/gallery
  Future<void> _pickFromCamera() async {
    final XFile? image = await _imagePicker.pickImage(
      source: ImageSource.camera,
      imageQuality: 80,
    );
    
    if (image != null) {
      final platformFile = PlatformFile(
        name: image.name,
        path: image.path,
        size: await File(image.path).length(),
      );
      setState(() => _selectedFiles = [platformFile]);
    }
  }

  Future<void> _pickFromGallery() async {
    final List<XFile> images = await _imagePicker.pickMultiImage(
      imageQuality: 80,
    );
    
    if (images.isNotEmpty) {
      List<PlatformFile> platformFiles = [];
      for (XFile image in images) {
        final platformFile = PlatformFile(
          name: image.name,
          path: image.path,
          size: await File(image.path).length(),
        );
        platformFiles.add(platformFile);
      }
      setState(() => _selectedFiles = platformFiles);
    }
  }

  Future<void> _uploadFiles() async {
    if (_selectedFiles.isEmpty) return;

    try {
      final apiService = Provider.of<ApiService>(context, listen: false);
      final result = await apiService.uploadBulkFiles(_selectedFiles);
      
      if (mounted) {
        final savedFiles = result['saved_files'] ?? 0;
        final skippedFiles = result['skipped_files'] ?? 0;
        final errors = result['errors'] as List? ?? [];
        final classDistribution = result['class_distribution'] as Map<String, dynamic>? ?? {};
        
        String message = 'Upload completed!\n';
        message += '✅ ${savedFiles} new files saved\n';
        if (skippedFiles > 0) {
          message += '⏭️ ${skippedFiles} duplicates skipped\n';
        }
        if (errors.isNotEmpty) {
          message += '❌ ${errors.length} errors occurred\n';
        }
        
        if (classDistribution.isNotEmpty) {
          message += '\n📊 Class Distribution:\n';
          classDistribution.forEach((className, count) {
            message += '• $className: $count images\n';
          });
        }
        
        showDialog(
          context: context,
          builder: (context) => AlertDialog(
            title: const Text('Upload Results'),
            content: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(message),
                  if (errors.isNotEmpty) ...[
                    const SizedBox(height: 16),
                    const Text('Errors:', style: TextStyle(fontWeight: FontWeight.bold, color: Colors.red)),
                    ...errors.map((error) => Text('• $error', style: const TextStyle(color: Colors.red))),
                  ],
                ],
              ),
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('OK'),
              ),
            ],
          ),
        );
        
        // Clear selected files after successful upload
        if (savedFiles > 0) {
          setState(() {
            _selectedFiles.clear();
          });
        }
        
        // Refresh database stats in API service
        await apiService.fetchDatabaseStats();
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Upload failed: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  // Add this method to help users rename files
  void _showRenameDialog(int index) {
    final TextEditingController controller = TextEditingController(
      text: _selectedFiles[index].name,
    );
    
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Rename File'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text('Add weather type prefix:'),
            const SizedBox(height: 8),
            TextField(
              controller: controller,
              decoration: const InputDecoration(
                hintText: 'e.g., Sunny_image1.jpg',
                labelText: 'Filename',
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              'Examples:\n• Sunny_photo1.jpg\n• Rainy_image2.jpg\n• Cloudy_pic3.jpg',
              style: TextStyle(fontSize: 12, color: Colors.grey),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () {
              setState(() {
                _selectedFiles[index] = PlatformFile(
                  name: controller.text,
                  path: _selectedFiles[index].path,
                  size: _selectedFiles[index].size,
                );
              });
              Navigator.pop(context);
            },
            child: const Text('Rename'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final apiService = Provider.of<ApiService>(context);

    return Scaffold(
      appBar: AppBar(title: const Text('Bulk Upload')),
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
                      'Upload Weather Images',
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'Name your images with weather type prefix:\n'
                      '• Shine_image1.jpg (or Sunny_image1.jpg)\n'
                      '• Rain_image1.jpg (or Rainy_image1.jpg)\n'
                      '• Cloudy_image1.jpg\n'
                      '• Sunrise_image1.jpg',
                      style: TextStyle(fontSize: 12, color: Colors.grey),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 16),
                    
                    // Multiple upload options
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      alignment: WrapAlignment.center,
                      children: [
                        ElevatedButton.icon(
                          onPressed: apiService.isLoading ? null : _pickFilesFromComputer,
                          icon: const Icon(Icons.folder_open),
                          label: const Text('From Computer'),
                        ),
                        ElevatedButton.icon(
                          onPressed: apiService.isLoading ? null : _pickFromGallery,
                          icon: const Icon(Icons.photo_library),
                          label: const Text('From Gallery'),
                        ),
                        ElevatedButton.icon(
                          onPressed: apiService.isLoading ? null : _pickFromCamera,
                          icon: const Icon(Icons.camera_alt),
                          label: const Text('Take Photo'),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            
            const SizedBox(height: 16),
            
            if (_selectedFiles.isNotEmpty) ...[
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Selected ${_selectedFiles.length} files',
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                      const SizedBox(height: 8),
                      SizedBox(
                        height: 120,
                        child: ListView.builder(
                          scrollDirection: Axis.horizontal,
                          itemCount: _selectedFiles.length,
                          itemBuilder: (context, index) {
                            final file = _selectedFiles[index];
                            return Padding(
                              padding: const EdgeInsets.all(4.0),
                              child: GestureDetector(
                                onTap: () => _showRenameDialog(index),
                                child: Container(
                                  width: 100,
                                  height: 100,
                                  decoration: BoxDecoration(
                                    color: Colors.grey[300],
                                    borderRadius: BorderRadius.circular(8),
                                    border: Border.all(color: Colors.grey),
                                  ),
                                  child: InkWell(
                                    onTap: () => _showRenameDialog(index),
                                    child: Column(
                                      mainAxisAlignment: MainAxisAlignment.center,
                                      children: [
                                        file.path != null
                                            ? ClipRRect(
                                                borderRadius: BorderRadius.circular(4),
                                                child: Image.file(
                                                  File(file.path!),
                                                  width: 80,
                                                  height: 60,
                                                  fit: BoxFit.cover,
                                                  errorBuilder: (context, error, stackTrace) {
                                                    return const Icon(Icons.image, size: 40);
                                                  },
                                                ),
                                              )
                                            : const Icon(Icons.image, size: 40),
                                        const SizedBox(height: 2),
                                        const Icon(Icons.edit, size: 12, color: Colors.blue),
                                      ],
                                    ),
                                  ),
                                ),
                              ),
                            );
                          },
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),
            ],
            
            if (apiService.isLoading)
              const Center(
                child: Column(
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 8),
                    Text('Uploading files...'),
                  ],
                ),
              )
            else
              FilledButton(
                onPressed: _selectedFiles.isEmpty ? null : _uploadFiles,
                child: Text(_selectedFiles.isEmpty 
                    ? 'Select Files First' 
                    : 'Upload ${_selectedFiles.length} Files'),
              ),
            
            if (apiService.message.isNotEmpty) ...[
              const SizedBox(height: 16),
              Text(
                apiService.message,
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: apiService.message.contains('Error') 
                      ? Colors.red 
                      : Colors.green,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}