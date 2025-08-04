import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:weather_classifier/screens/dashboard.dart';
import 'package:weather_classifier/screens/predict.dart';
import 'package:weather_classifier/screens/retrain.dart';
import 'package:weather_classifier/screens/upload.dart';
import 'package:weather_classifier/screens/database.dart';
import 'package:weather_classifier/services/api_service.dart';

void main() {
  runApp(
    ChangeNotifierProvider(
      create: (context) => ApiService(baseUrl: 'http://localhost:8000'), // Change to port 8000
      child: const WeatherClassifierApp(),
    ),
  );
}

class WeatherClassifierApp extends StatelessWidget {
  const WeatherClassifierApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Weather Classifier',
      theme: ThemeData(
        colorScheme: const ColorScheme.light(
          primary: Color(0xFF1E88E5),
          secondary: Color(0xFFFF9800),
          surface: Color(0xFFF8F9FA),
          background: Color(0xFFE3F2FD),
          onPrimary: Colors.white,
          onSecondary: Colors.white,
        ),
        useMaterial3: true,
        cardTheme: CardThemeData(
          elevation: 8,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF1E88E5),
          foregroundColor: Colors.white,
          elevation: 0,
        ),
      ),
      home: const AppScaffold(),
    );
  }
}

class AppScaffold extends StatefulWidget {
  const AppScaffold({super.key});

  @override
  State<AppScaffold> createState() => _AppScaffoldState();
}

class _AppScaffoldState extends State<AppScaffold> {
  int _currentIndex = 0;

  final List<Widget> _screens = [
    DashboardScreen(key: DashboardScreen.globalKey),
    const PredictScreen(),
    const UploadScreen(),
    const DatabaseScreen(),
    const RetrainScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _screens[_currentIndex],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: (index) {
          setState(() => _currentIndex = index);
          // Refresh dashboard when navigating to it, especially after retraining
          if (index == 0) { // Dashboard is at index 0
            DashboardScreen.refreshDashboard();
          }
        },
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.dashboard),
            label: 'Dashboard',
          ),
          NavigationDestination(
            icon: Icon(Icons.cloud),
            label: 'Predict',
          ),
          NavigationDestination(
            icon: Icon(Icons.upload_file),
            label: 'Upload',
          ),
          NavigationDestination(
            icon: Icon(Icons.storage),
            label: 'Database',
          ),
          NavigationDestination(
            icon: Icon(Icons.model_training),
            label: 'Retrain',
          ),
        ],
      ),
    );
  }
}