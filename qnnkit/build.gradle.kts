plugins {
  id("com.android.library")
  id("maven-publish")
}

group = "io.qnnkit"
version = "0.1.0"

android {
  namespace = "io.qnnkit"
  compileSdk = 36
  ndkVersion = "27.0.12077973"

  defaultConfig {
    minSdk = 30

    ndk {
      abiFilters += "arm64-v8a"
    }
    externalNativeBuild {
      cmake {
        cppFlags += "-std=c++17"
      }
    }
  }

  externalNativeBuild {
    cmake {
      path = file("src/main/cpp/CMakeLists.txt")
      version = "3.22.1"
    }
  }

  buildTypes {
    named("release") {
      isMinifyEnabled = false
    }
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }

  publishing {
    singleVariant("release")
  }
}

publishing {
  publications {
    register<MavenPublication>("release") {
      groupId = "io.qnnkit"
      artifactId = "qnnkit"
      version = "0.1.0"
      afterEvaluate { from(components["release"]) }
    }
  }
}

// The QNN SDK content is not committed; fail early with a clear message if the
// one-time fetch hasn't run yet.
val checkQnnSdk = tasks.register("checkQnnSdk") {
  doFirst {
    val header = file("src/main/cpp/include/QNN/QnnInterface.h")
    val lib = file("src/main/jniLibs/arm64-v8a/libQnnHtp.so")
    if (!header.exists() || !lib.exists()) {
      throw GradleException(
        "QNN SDK content missing. Run ./tools/fetch-qnn-sdk.sh first " +
        "(needs a local Qualcomm QAIRT SDK; set QNN_SDK_ROOT). See README.md.")
    }
  }
}
tasks.named("preBuild") { dependsOn(checkQnnSdk) }
