# qnnkit

Reusable raw-QNN inference for Android. Run your own models on the Qualcomm Hexagon
NPU, fully offline, inside your app process. No server, no NNAPI, no extra runtime:
just the bundled Qualcomm QNN libraries and a thin JNI layer with a small Java API.

It exists because the hard parts of talking to the Hexagon from an ordinary app are
already solved here: reaching the DSP in-process, FP16 graphs on the HTP, loading a
QAIRT DLC via `composeGraphsFromDlc`, caching the finalized graph as a context binary,
and freeing the per-call handles so a second run does not run out of NPU memory.

## What you get

- `io.qnnkit.Npu` - load a model and check availability.
- `io.qnnkit.QnnModel` - `float[] run(float[] input)` on the NPU.
- `io.qnnkit.SocInfo` - runtime SoC detection (Qualcomm? which chip? which QNN SoC id?).
- All the QNN runtime `.so` (HTP + skels V68..V79 + System + ModelDlc) bundled in the AAR,
  plus the `uses-native-library` / `extractNativeLibs` manifest entries your app inherits.

## Requirements

- A Qualcomm Snapdragon device with a Hexagon NPU (arm64-v8a). Not Exynos, MediaTek or Tensor.
- `minSdk 30` in the consumer (API 31+ is where vendor native libs get declared).
- Your model already converted for QNN:
  - a QAIRT DLC (`qairt-converter`), loaded with `Npu.loadDlc(...)`, or
  - a QNN model `.so` (`qnn-onnx-converter` + `qnn-model-lib-generator`), shipped in your
    app `jniLibs/arm64-v8a/` and loaded with `Npu.loadSo(...)`.

## Build

This repo holds only the qnnkit code (MIT). The Qualcomm QNN pieces it links against
(headers, SampleApp helper sources, and the `libQnn*.so` runtime) are proprietary and are
NOT committed here. Fetch them once from a local Qualcomm AI Engine Direct (QAIRT) SDK,
which is a free download from Qualcomm:

```bash
# point at your SDK, or let it auto-detect /opt/qcom/aistack/qairt/*
QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.31.0.250130 ./tools/fetch-qnn-sdk.sh
```

Then build the AAR and publish it to your local Maven:

```bash
./gradlew :qnnkit:publishToMavenLocal        # or :qnnkit:assembleRelease for the .aar
```

The build fails early with a clear message if the fetch step has not run.

```kotlin
// settings.gradle.kts  ->  dependencyResolutionManagement { repositories { mavenLocal() } }
// app/build.gradle.kts
dependencies {
  implementation("io.qnnkit:qnnkit:0.1.0")
}

// Keep native libs extracted to disk so the CDSP can load the Hexagon skel:
android { packaging { jniLibs { useLegacyPackaging = true } } }
```

The `.aar` is also usable as a plain file dependency if you prefer not to publish.

## Use

```java
if (Npu.isAvailable(ctx)) {
  // detector.dlc is in assets/ ; graph name is the one in the "finalize=... graph=..." log
  QnnModel detector   = Npu.loadDlc(ctx, "detector.dlc", "detector_qairt");
  QnnModel recognizer = Npu.loadSo(ctx, "librecognizer.so", "recognizer");

  float[] heat = detector.run(inputFloats);     // NPU, offline
  // ... your pre/post-processing ...
}
```

First `run` composes and finalizes on the NPU (a few seconds) and writes a context
binary next to the cache dir; later runs load that and skip the finalize. The cache is
keyed by the model file size, so swapping a model file re-finalizes automatically.

## SoC support

`SocInfo.detect()` reads `Build.SOC_MODEL` and maps known chips to a QNN SoC id. Only
`SM8450` (Snapdragon 8 Gen 1, Hexagon v69) is verified today; other Snapdragons fall back
to id `0`, which lets the HTP backend detect the attached chip on-device. To pin a chip
explicitly, add it to the `mapSoc` table in `SocInfo.java` from the `SocModel` enum in the
QNN SDK's `QnnHtpDevice.h`.

## Persistent sessions over prebuilt context binaries (0.2.0)

For models called many times in a row (TTS vocoders, Whisper decoders), compile the context
binary offline for the target SoC with `qnn-context-binary-generator` (same QAIRT version as the
runtime you bundle) and keep it loaded:

```java
try (QnnSession dec = Npu.openContext(ctx, new File(dir, "decoder.bin"))) {
  dec.input("input_ids").asIntBuffer().put(0, token);   // any number of named inputs
  if (dec.execute()) dec.output("logits").asShortBuffer(); // any number of outputs
}
```

- Inputs and outputs are direct `ByteBuffer`s over the tensors' own memory (native order),
  of any QNN data type: int32, fp32, fp16 (as AI Hub exports), uint8...
- One backend + device is shared by all sessions in the process; each `execute()` only runs
  the graph. `QnnSession.setBurst(true/false)` votes the HTP clocks around a burst of work.
- `QnnSession.lastError()` says which QNN step failed and with which error code.
- Load is a memory map of the file, so hundreds of MB do not pass through the Java heap.

## Scope and limits

- `QnnModel.run` (DLC / model .so): one input tensor and one output tensor per call, handles
  created and freed per call. Use `QnnSession` for multi input/output or repeated calls.
- arm64-v8a only. Float graphs run in FP16 on the HTP.

## License

qnnkit's own code is **MIT** (`cpp/qnn_jni.cpp`, `cpp/CMakeLists.txt`, the `io.qnnkit` Java,
the Gradle build, `tools/`). See [LICENSE](LICENSE).

The Qualcomm QNN content that `tools/fetch-qnn-sdk.sh` pulls in (QNN headers, SampleApp helper
sources, `libQnn*.so`) is "Confidential and Proprietary - Qualcomm Technologies, Inc." and is
governed by the QAIRT SDK license, not MIT. It is deliberately not committed here, so this repo
is safe to publish. The AAR you build locally does bundle those libraries: shipping or
distributing that AAR (or an app built from it) is subject to Qualcomm's SDK license, so check
its terms before you do.
