# Weather Classification ML Pipeline

A comprehensive machine learning application for weather classification using Flutter frontend and FastAPI backend with TensorFlow/Keras models.

## 🌟 Features

- **Real-time Weather Classification**: Classify weather images into 4 categories (Cloudy, Rain, Shine, Sunrise)
- **Interactive Dashboard**: Visualize model performance and training metrics
- **Smart Retraining**: Transfer learning approach that preserves model knowledge
- **Performance Monitoring**: Training charts and class distribution analysis
- **Mobile-Ready**: Flutter frontend with responsive design

## 🏗️ Architecture

### Backend (FastAPI + TensorFlow)
- **ML Model**: CNN with 85.8% accuracy on weather classification
- **Smart Training**: Transfer learning with feature extraction
- **API Endpoints**: Prediction, retraining, health checks
- **Model Management**: Automatic model loading and fallback strategies

### Frontend (Flutter)
- **Dashboard**: Training performance visualization with fl_chart
- **Prediction Interface**: Image upload and real-time classification
- **Upload Management**: Batch image upload for dataset expansion
- **Retraining Controls**: One-click model improvement

## 📊 Model Performance

- **Base Accuracy**: 85.8% on validation set
- **Classes**: 4 weather categories
- **Architecture**: Convolutional Neural Network
- **Training Strategy**: Transfer learning with smart feature extraction

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Flutter SDK
- TensorFlow 2.x

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend Setup
```bash
cd frontend
flutter pub get
flutter run
```

## 📱 Screenshots

### Dashboard
- Training performance charts
- Model accuracy visualization
- Class distribution analysis

### Prediction Interface
- Drag-and-drop image upload
- Real-time classification results
- Confidence scores

## 🔧 Technical Details

### Smart Training Strategy
The application implements a sophisticated retraining approach:

1. **Feature Extraction**: Preserves learned features from base model
2. **Transfer Learning**: Fine-tunes only the classification layers
3. **Performance Monitoring**: Tracks accuracy improvements
4. **Automatic Fallback**: Uses best-performing model for predictions

### Model Management
- Automatic model loading with prioritization
- Health checks and error handling
- Performance tracking and persistence

## 📈 Performance Improvements

Recent optimizations have achieved:
- **Model Preservation**: Maintained 85.8% base accuracy during retraining
- **Training Efficiency**: 15.6% → 80% accuracy recovery through smart training
- **User Experience**: Real-time dashboard updates and performance visualization

## 🛠️ Development

### Project Structure
```
├── backend/           # FastAPI ML API
│   ├── main.py       # Main API server
│   ├── retrain.py    # Training logic
│   └── requirements.txt
├── frontend/         # Flutter application
│   ├── lib/
│   │   ├── screens/  # App screens
│   │   ├── services/ # API services
│   │   └── widgets/  # Custom widgets
└── weather_dataset/  # Training data
```

### Key Components
- **API Service**: Flutter service for backend communication
- **Training Charts**: Interactive performance visualization
- **Model Handler**: Smart model loading and management
- **Upload Manager**: Batch image processing

## 🚦 Testing

The project includes comprehensive testing capabilities:
- Unit tests for API endpoints
- Widget tests for Flutter components
- Performance monitoring and metrics

## 📝 License

This project is created for educational purposes as part of an ML pipeline demonstration.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 🔗 Related Documentation

- [TensorFlow Documentation](https://tensorflow.org)
- [Flutter Documentation](https://flutter.dev)
- [FastAPI Documentation](https://fastapi.tiangolo.com)

---

**Built with ❤️ using Flutter, FastAPI, and TensorFlow**
