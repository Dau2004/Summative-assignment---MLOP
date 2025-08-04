import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';

class TrainingPerformanceChart extends StatelessWidget {
  final Map<String, dynamic>? trainingData;

  const TrainingPerformanceChart({
    Key? key,
    required this.trainingData,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    if (trainingData == null) {
      return const Center(
        child: Text(
          'No training data available',
          style: TextStyle(fontSize: 14, color: Colors.grey),
        ),
      );
    }

    return Column(
      children: [
        // Legend
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            Expanded(child: _buildLegendItem('Train Acc', Colors.blue, false)),
            Expanded(child: _buildLegendItem('Val Acc', Colors.green, true)),
            Expanded(child: _buildLegendItem('Loss (scaled)', Colors.red, false)),
          ],
        ),
        const SizedBox(height: 16),
        // Chart
        SizedBox(
          height: 200,
          child: LineChart(
            LineChartData(
              gridData: FlGridData(show: true),
              titlesData: FlTitlesData(
                leftTitles: AxisTitles(
                  sideTitles: SideTitles(
                    showTitles: true,
                    reservedSize: 40,
                    getTitlesWidget: (value, meta) {
                      return Text(
                        value.toStringAsFixed(1),
                        style: const TextStyle(fontSize: 10),
                      );
                    },
                  ),
                ),
                bottomTitles: AxisTitles(
                  sideTitles: SideTitles(
                    showTitles: true,
                    reservedSize: 30,
                    getTitlesWidget: (value, meta) {
                      return Text(
                        'E${value.toInt()}',
                        style: const TextStyle(fontSize: 10),
                      );
                    },
                  ),
                ),
                topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
              ),
              borderData: FlBorderData(show: true),
              lineBarsData: _getLineBarsData(),
              minY: 0,
              maxY: 1,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildLegendItem(String label, Color color, bool isDashed) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 20,
          height: 2,
          decoration: BoxDecoration(
            color: color,
            border: isDashed ? Border.all(color: color) : null,
          ),
          child: isDashed
              ? CustomPaint(
                  painter: DashedLinePainter(color: color),
                )
              : null,
        ),
        const SizedBox(width: 4),
        Expanded(
          child: Text(
            label,
            style: const TextStyle(fontSize: 10),
            overflow: TextOverflow.ellipsis,
            maxLines: 1,
          ),
        ),
      ],
    );
  }

  List<LineChartBarData> _getLineBarsData() {
    final accuracyHistory = List<double>.from(trainingData?['accuracy_history'] ?? []);
    final valAccuracyHistory = List<double>.from(trainingData?['val_accuracy_history'] ?? []);
    final lossHistory = List<double>.from(trainingData?['loss_history'] ?? []);

    List<LineChartBarData> lines = [];

    // Training Accuracy Line
    if (accuracyHistory.isNotEmpty) {
      lines.add(
        LineChartBarData(
          spots: accuracyHistory.asMap().entries.map((entry) {
            return FlSpot(entry.key.toDouble(), entry.value);
          }).toList(),
          isCurved: true,
          color: Colors.blue,
          barWidth: 2,
          dotData: const FlDotData(show: true),
        ),
      );
    }

    // Validation Accuracy Line
    if (valAccuracyHistory.isNotEmpty) {
      lines.add(
        LineChartBarData(
          spots: valAccuracyHistory.asMap().entries.map((entry) {
            return FlSpot(entry.key.toDouble(), entry.value);
          }).toList(),
          isCurved: true,
          color: Colors.green,
          barWidth: 2,
          dotData: const FlDotData(show: true),
          dashArray: [5, 5], // Dashed line for validation
        ),
      );
    }

    // Training Loss Line (scaled to 0-1 range for visibility)
    if (lossHistory.isNotEmpty) {
      final maxLoss = lossHistory.reduce((a, b) => a > b ? a : b);
      lines.add(
        LineChartBarData(
          spots: lossHistory.asMap().entries.map((entry) {
            // Scale loss to 0-1 range for better visualization
            final scaledLoss = maxLoss > 0 ? entry.value / maxLoss : 0.0;
            return FlSpot(entry.key.toDouble(), scaledLoss);
          }).toList(),
          isCurved: true,
          color: Colors.red,
          barWidth: 2,
          dotData: const FlDotData(show: true),
        ),
      );
    }

    return lines;
  }
}

class DashedLinePainter extends CustomPainter {
  final Color color;

  DashedLinePainter({required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = 2;

    const dashWidth = 3.0;
    const dashSpace = 3.0;
    double startX = 0;

    while (startX < size.width) {
      canvas.drawLine(
        Offset(startX, size.height / 2),
        Offset(startX + dashWidth, size.height / 2),
        paint,
      );
      startX += dashWidth + dashSpace;
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}
