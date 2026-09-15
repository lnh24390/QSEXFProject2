# ultralytics_yolo (LiteRT 2.x) 릴리즈 빌드에서 모델 로딩 크래시 방지
-keep class com.google.ai.edge.litert.** { *; }
-keep interface com.google.ai.edge.litert.** { *; }
-dontwarn com.google.ai.edge.litert.**
-keep class org.tensorflow.** { *; }
-keep class com.ultralytics.** { *; }
-dontwarn org.tensorflow.**

# androidx.work(+Room): WorkDatabase_Impl 을 리플렉션으로 생성하므로 R8 이 지우면
# androidx.startup.InitializationProvider 단계에서 앱이 실행 즉시 죽는다.
#   java.lang.RuntimeException: Failed to create an instance of class androidx.work.impl.WorkDatabase
-keep class * extends androidx.room.RoomDatabase { *; }
-keep @androidx.room.Database class * { *; }
-keep class androidx.work.impl.** { *; }
-dontwarn androidx.work.**
