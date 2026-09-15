import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'screens/detection_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  runApp(const TrashSorterApp());
}

class TrashSorterApp extends StatelessWidget {
  const TrashSorterApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '분리수거 도우미',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF2E7D32),
          brightness: Brightness.dark,
        ),
        scaffoldBackgroundColor: Colors.black,
      ),
      home: const DetectionScreen(),
    );
  }
}
