allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

// permission_handler_android 14.1.0 은 compileSdk 37 을 요구한다(AAR 메타데이터의 minCompileSdk).
// 그런데 Google 이 배포하는 플랫폼 패키지는 android-37.0 / 37.1 뿐이라, minor 를 지정하지 않으면
// AGP 가 존재하지 않는 'android-37' 해시를 찾다가 실패한다.
// AGP 9.1 의 compileSdkMinor 로 android-37.0 을 가리키게 해서 앱·플러그인 모듈을 함께 맞춘다.
// (Flutter 기본값 flutter.compileSdkVersion 은 아직 36 이라 그대로 두면 AAR 검사에서 걸린다.)
val projectCompileSdk = 37
val projectCompileSdkMinor = 0

// :app 은 위의 evaluationDependsOn 때문에 먼저 평가되므로 여기서 덮을 수 없다
// ("It is too late to set compileSdk"). 앱 모듈은 app/build.gradle.kts 에서 직접 지정한다.
subprojects {
    plugins.withId("com.android.library") {
        extensions.configure<com.android.build.api.dsl.LibraryExtension>("android") {
            compileSdk = projectCompileSdk
            compileSdkMinor = projectCompileSdkMinor
        }
    }
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
