#!/usr/bin/env bash
#
# Populate the Qualcomm QNN bits that qnnkit needs to build. These are NOT
# redistributed in this repo (they are "Confidential and Proprietary" to Qualcomm);
# this script copies them from a local Qualcomm AI Engine Direct (QAIRT) SDK, which
# is a free download from Qualcomm. Run it once after cloning, then build.
#
#   QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.31.0.250130 ./tools/fetch-qnn-sdk.sh
#
# If QNN_SDK_ROOT is unset it auto-detects the newest /opt/qcom/aistack/qairt/*.
# Tested against QAIRT 2.31.x (Hexagon v68..v79). Other versions should work but
# are not guaranteed.

set -euo pipefail

SDK="${QNN_SDK_ROOT:-${QAIRT_SDK_ROOT:-}}"
if [ -z "$SDK" ]; then
  SDK="$(ls -d /opt/qcom/aistack/qairt/*/ 2>/dev/null | sort -V | tail -1 || true)"
fi
SDK="${SDK%/}"
if [ -z "$SDK" ] || [ ! -d "$SDK/include/QNN" ]; then
  echo "error: QNN SDK not found." >&2
  echo "  Install the Qualcomm AI Engine Direct (QAIRT) SDK and point QNN_SDK_ROOT at it:" >&2
  echo "  QNN_SDK_ROOT=/path/to/qairt/<version> ./tools/fetch-qnn-sdk.sh" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN="$REPO_ROOT/qnnkit/src/main"
CPP="$MAIN/cpp"
JNI="$MAIN/jniLibs/arm64-v8a"
SRC="$SDK/examples/QNN/SampleApp/SampleApp/src"

echo "qnnkit: fetching from $SDK"

# 1) QNN headers.
rm -rf "$CPP/include/QNN"
mkdir -p "$CPP/include"
cp -r "$SDK/include/QNN" "$CPP/include/QNN"

# 2) SampleApp helper sources (tensor setup, dynamic load, logging, PAL).
rm -rf "$CPP/qnn"
mkdir -p "$CPP/qnn"
for d in Utils WrapperUtils Log PAL DspLog; do
  cp -r "$SRC/$d" "$CPP/qnn/$d"
done
cp "$SRC/SampleApp.hpp" "$SRC/QnnTypeMacros.hpp" "$CPP/qnn/"

# 3) Runtime libraries (arm64-v8a host stubs + Hexagon skels).
mkdir -p "$JNI"
for l in libQnnHtp.so libQnnHtpPrepare.so libQnnSystem.so libQnnModelDlc.so \
         libQnnHtpV68Stub.so libQnnHtpV69Stub.so libQnnHtpV73Stub.so \
         libQnnHtpV75Stub.so libQnnHtpV79Stub.so; do
  cp "$SDK/lib/aarch64-android/$l" "$JNI/"
done
for v in 68 69 73 75 79; do
  cp "$SDK/lib/hexagon-v$v/unsigned/libQnnHtpV${v}Skel.so" "$JNI/"
done
cp "$SDK/lib/hexagon-v73/unsigned/libQnnHtpV73.so" "$JNI/"

echo "qnnkit: done. Headers, SampleApp sources and $(ls "$JNI" | wc -l) libs in place."
echo "        Now build:  ./gradlew :qnnkit:assembleRelease"
