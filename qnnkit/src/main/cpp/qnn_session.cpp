// qnnkit - persistent sessions over prebuilt QNN context binaries.
//
// JNI backing io.qnnkit.QnnSession. Where QnnNative.run() composes a model and tears every
// handle down per call, a session loads a context binary once (compiled offline for the
// target SoC, e.g. with qnn-context-binary-generator) and keeps it on the NPU, so repeated
// calls only pay for graphExecute. Graphs may have any number of inputs and outputs of any
// data type: each tensor's client buffer is exposed to Java as a direct ByteBuffer, so data
// moves without extra copies.
//
// One process-wide runtime (backend + device + log + power vote) is shared by all sessions;
// it is created with the first session and freed with the last one.

#include <jni.h>
#include <android/log.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "QNN/QnnInterface.h"
#include "QNN/QnnBackend.h"
#include "QNN/QnnContext.h"
#include "QNN/QnnDevice.h"
#include "QNN/QnnGraph.h"
#include "QNN/QnnLog.h"
#include "QNN/HTP/QnnHtpDevice.h"
#include "QNN/HTP/QnnHtpPerfInfrastructure.h"
#include "QNN/System/QnnSystemContext.h"

#include "SampleApp.hpp"
#include "Utils/DynamicLoadUtil.hpp"
#include "Utils/IOTensor.hpp"
#include "Utils/QnnSampleAppUtils.hpp"
#include "WrapperUtils/QnnWrapperUtils.hpp"
#include "QnnTypeMacros.hpp"

using namespace qnn::tools;

#define TAG "qnnkit"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

void qnnLog(const char* fmt, QnnLog_Level_t, uint64_t, va_list ap) {
  char b[1024];
  vsnprintf(b, sizeof(b), fmt, ap);
  __android_log_print(ANDROID_LOG_INFO, "QnnLib", "%s", b);
}

struct Runtime {
  sample_app::QnnFunctionPointers qfp{};
  void* backendLib = nullptr;
  Qnn_LogHandle_t log = nullptr;
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;
  QnnHtpDevice_PerfInfrastructure_t* perf = nullptr;
  uint32_t powerId = 0;
  bool hasPower = false;
  int sessions = 0;
};

std::mutex gLock;
Runtime* gRt = nullptr;
std::string gLastError;  // why the last nativeOpen failed, for QnnSession.lastError()

void fail(const char* fmt, ...) {
  char b[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(b, sizeof(b), fmt, ap);
  va_end(ap);
  gLastError = b;
  LOGE("%s", b);
}

struct Session {
  Qnn_ContextHandle_t context = nullptr;
  qnn_wrapper_api::GraphInfo_t** graphs = nullptr;
  uint32_t numGraphs = 0;
  Qnn_Tensor_t* inputs = nullptr;
  Qnn_Tensor_t* outputs = nullptr;
  iotensor::IOTensor io;
};

// Same device setup as qnn_jni.cpp: unsigned process domain (no OEM signature needed),
// plus an explicit SoC when known.
bool createDevice(Runtime* rt, int socModel) {
  QnnHtpDevice_CustomConfig_t soc{}, pd{};
  QnnDevice_Config_t cSoc{}, cPd{};
  std::vector<const QnnDevice_Config_t*> cfgs;
  pd.option = QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD;
  pd.useSignedProcessDomain.deviceId = 0;
  pd.useSignedProcessDomain.useSignedProcessDomain = false;
  cPd.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
  cPd.customConfig = &pd;
  if (socModel > 0) {
    soc.option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
    soc.socModel = (uint32_t)socModel;
    cSoc.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
    cSoc.customConfig = &soc;
    cfgs.push_back(&cSoc);
  }
  cfgs.push_back(&cPd);
  cfgs.push_back(nullptr);
  return rt->qfp.qnnInterface.deviceCreate(rt->log, cfgs.data(), &rt->device) == QNN_SUCCESS;
}

void freeRuntime(Runtime* rt) {
  auto& itf = rt->qfp.qnnInterface;
  if (rt->hasPower && rt->perf) rt->perf->destroyPowerConfigId(rt->powerId);
  if (rt->device && itf.deviceFree) itf.deviceFree(rt->device);
  if (rt->backend && itf.backendFree) itf.backendFree(rt->backend);
  if (rt->log && itf.logFree) itf.logFree(rt->log);
  delete rt;
}

Runtime* acquireRuntime(const char* libdir, int socModel) {
  if (gRt) {
    gRt->sessions++;
    return gRt;
  }
  setenv("ADSP_LIBRARY_PATH", libdir, 1);
  // The app linker namespace won't resolve the vendor FastRPC lib by soname.
  dlopen("/vendor/lib64/libcdsprpc.so", RTLD_NOW | RTLD_GLOBAL);

  auto* rt = new Runtime();
  std::string dir(libdir);
  if (dynamicloadutil::getQnnFunctionPointers(dir + "/libQnnHtp.so", "", &rt->qfp, &rt->backendLib, false, nullptr)
          != dynamicloadutil::StatusCode::SUCCESS ||
      dynamicloadutil::getQnnSystemFunctionPointers(dir + "/libQnnSystem.so", &rt->qfp)
          != dynamicloadutil::StatusCode::SUCCESS) {
    fail("session: loading libQnnHtp/libQnnSystem failed");
    delete rt;
    return nullptr;
  }
  auto& itf = rt->qfp.qnnInterface;
  itf.logCreate(qnnLog, QNN_LOG_LEVEL_ERROR, &rt->log);
  if (itf.backendCreate(rt->log, nullptr, &rt->backend) != QNN_SUCCESS || !createDevice(rt, socModel)) {
    fail("session: backend/device create failed");
    freeRuntime(rt);
    return nullptr;
  }
  QnnDevice_Infrastructure_t infra = nullptr;
  if (itf.deviceGetInfrastructure && itf.deviceGetInfrastructure(&infra) == QNN_SUCCESS && infra) {
    auto* htp = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(infra);
    if (htp->infraType == QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) {
      rt->perf = &htp->perfInfra;
      rt->hasPower = rt->perf->createPowerConfigId(0, 0, &rt->powerId) == QNN_SUCCESS;
    }
  }
  rt->sessions = 1;
  gRt = rt;
  return rt;
}

void releaseRuntime() {
  if (gRt && --gRt->sessions == 0) {
    freeRuntime(gRt);
    gRt = nullptr;
  }
}

// Burst while work is pending (DCVS off, max voltage corners, short sleep latency);
// otherwise hand the clocks back to DCVS so an idle keyboard does not drain the battery.
bool setPower(Runtime* rt, bool burst) {
  if (!rt->hasPower) return false;
  QnnHtpPerfInfrastructure_PowerConfig_t dcvs;
  memset(&dcvs, 0, sizeof(dcvs));
  dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  auto& d = dcvs.dcvsV3Config;
  d.contextId = rt->powerId;
  d.setDcvsEnable = 1;
  d.dcvsEnable = burst ? 0 : 1;
  d.powerMode = burst ? QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE
                      : QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;
  d.setSleepLatency = 1;
  d.sleepLatency = burst ? 40 : 1000;
  d.setSleepDisable = 1;
  d.sleepDisable = burst ? 1 : 0;
  d.setBusParams = 1;
  d.setCoreParams = 1;
  auto hi = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  auto lo = DCVS_VOLTAGE_VCORNER_SVS;
  d.busVoltageCornerMin = burst ? hi : lo;
  d.busVoltageCornerTarget = burst ? hi : lo;
  d.busVoltageCornerMax = hi;
  d.coreVoltageCornerMin = burst ? hi : lo;
  d.coreVoltageCornerTarget = burst ? hi : lo;
  d.coreVoltageCornerMax = hi;

  QnnHtpPerfInfrastructure_PowerConfig_t rpc;
  memset(&rpc, 0, sizeof(rpc));
  rpc.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
  rpc.rpcPollingTimeConfig = burst ? 9999 : 0;

  const QnnHtpPerfInfrastructure_PowerConfig_t* cfg[] = {&dcvs, &rpc, nullptr};
  return rt->perf->setPowerConfig(rt->powerId, cfg) == QNN_SUCCESS;
}

size_t dataTypeSize(Qnn_DataType_t t) {
  switch (t) {
    case QNN_DATATYPE_INT_8: case QNN_DATATYPE_UINT_8: case QNN_DATATYPE_BOOL_8:
    case QNN_DATATYPE_SFIXED_POINT_8: case QNN_DATATYPE_UFIXED_POINT_8:
      return 1;
    case QNN_DATATYPE_INT_16: case QNN_DATATYPE_UINT_16: case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_SFIXED_POINT_16: case QNN_DATATYPE_UFIXED_POINT_16:
      return 2;
    case QNN_DATATYPE_INT_32: case QNN_DATATYPE_UINT_32: case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_SFIXED_POINT_32: case QNN_DATATYPE_UFIXED_POINT_32:
      return 4;
    case QNN_DATATYPE_INT_64: case QNN_DATATYPE_UINT_64: case QNN_DATATYPE_FLOAT_64:
      return 8;
    default:
      return 0;
  }
}

// Client-side tensors for a retrieved graph: a deep copy of each tensor's info plus a zeroed
// RAW buffer sized from its dims and data type. Replaces SampleApp's IOTensor setup, whose
// allocator has no fp16 case (AI Hub exports fp16 I/O); teardown still goes through IOTensor,
// which frees with free() exactly what is allocated here.
bool setupTensors(Qnn_Tensor_t** out, uint32_t count, const Qnn_Tensor_t* wrappers) {
  *out = static_cast<Qnn_Tensor_t*>(calloc(count ? count : 1, sizeof(Qnn_Tensor_t)));
  if (!*out) return false;
  for (uint32_t i = 0; i < count; i++) {
    Qnn_Tensor_t* t = *out + i;
    *t = QNN_TENSOR_INIT;
    if (!sample_app::deepCopyQnnTensorInfo(t, &wrappers[i])) return false;
    QNN_TENSOR_SET_MEM_TYPE(t, QNN_TENSORMEMTYPE_RAW);
    size_t bytes = dataTypeSize(QNN_TENSOR_GET_DATA_TYPE(t));
    if (bytes == 0) {
      fail("session: tensor %s has unsupported data type 0x%x", QNN_TENSOR_GET_NAME(t),
           (unsigned)QNN_TENSOR_GET_DATA_TYPE(t));
      return false;
    }
    for (uint32_t d = 0; d < QNN_TENSOR_GET_RANK(t); d++) bytes *= QNN_TENSOR_GET_DIMENSIONS(t)[d];
    Qnn_ClientBuffer_t buf = QNN_CLIENT_BUFFER_INIT;
    buf.data = calloc(1, bytes);
    buf.dataSize = (uint32_t)bytes;
    QNN_TENSOR_SET_CLIENT_BUF(t, buf);
    if (!buf.data) return false;
  }
  return true;
}

Session* fromHandle(jlong h) { return reinterpret_cast<Session*>(h); }

const Qnn_Tensor_t& tensorAt(Session* s, bool output, int i) {
  return output ? s->outputs[i] : s->inputs[i];
}

uint32_t tensorCount(Session* s, bool output) {
  auto& g = (*s->graphs)[0];
  return output ? g.numOutputTensors : g.numInputTensors;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_io_qnnkit_QnnSession_nativeOpen(JNIEnv* env, jclass, jstring jlibdir, jstring jpath, jint socModel) {
  const char* libdir = env->GetStringUTFChars(jlibdir, nullptr);
  const char* path = env->GetStringUTFChars(jpath, nullptr);
  std::lock_guard<std::mutex> lock(gLock);
  gLastError.clear();
  Session* s = nullptr;
  Runtime* rt = acquireRuntime(libdir, socModel);
  int fd = -1;
  void* map = MAP_FAILED;
  size_t size = 0;
  QnnSystemContext_Handle_t sys = nullptr;
  do {
    if (!rt) break;
    fd = open(path, O_RDONLY);
    struct stat st{};
    if (fd < 0 || fstat(fd, &st) != 0 || st.st_size <= 0) {
      fail("session: cannot read %s", path);
      break;
    }
    size = (size_t)st.st_size;
    // Map instead of reading: context binaries are hundreds of MB and only needed until created.
    map = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) break;
    s = new Session();
    const QnnSystemContext_BinaryInfo_t* info = nullptr;
    Qnn_ContextBinarySize_t infoSize = 0;
    auto& sysItf = rt->qfp.qnnSystemInterface;
    auto& itf = rt->qfp.qnnInterface;
    Qnn_ErrorHandle_t e = QNN_SUCCESS;
    const char* step = nullptr;
    if (sysItf.systemContextCreate(&sys) != QNN_SUCCESS) step = "systemContextCreate";
    else if ((e = sysItf.systemContextGetBinaryInfo(sys, map, size, &info, &infoSize)) != QNN_SUCCESS) step = "getBinaryInfo";
    else if (!sample_app::copyMetadataToGraphsInfo(info, s->graphs, s->numGraphs) || s->numGraphs == 0) step = "copyMetadata";
    else if ((e = itf.contextCreateFromBinary(rt->backend, rt->device, nullptr, map, size, &s->context, nullptr)) != QNN_SUCCESS) step = "contextCreateFromBinary";
    else if ((e = itf.graphRetrieve(s->context, (*s->graphs)[0].graphName, &(*s->graphs)[0].graph)) != QNN_SUCCESS) step = "graphRetrieve";
    else if (!setupTensors(&s->inputs, (*s->graphs)[0].numInputTensors, (*s->graphs)[0].inputTensors) ||
             !setupTensors(&s->outputs, (*s->graphs)[0].numOutputTensors, (*s->graphs)[0].outputTensors)) step = "setupTensors";
    if (step) {
      if (gLastError.empty())
        fail("session: %s failed (err %llu) for %s (%zu bytes)", step, (unsigned long long)e, path, size);
      if (s->inputs || s->outputs)
        s->io.tearDownInputAndOutputTensors(s->inputs, s->outputs,
                                            s->inputs ? (*s->graphs)[0].numInputTensors : 0,
                                            s->outputs ? (*s->graphs)[0].numOutputTensors : 0);
      if (s->context) itf.contextFree(s->context, nullptr);
      delete s;
      s = nullptr;
      break;
    }
    LOGI("session: opened %s graph=%s in=%u out=%u", path, (*s->graphs)[0].graphName,
         tensorCount(s, false), tensorCount(s, true));
  } while (false);
  if (sys) rt->qfp.qnnSystemInterface.systemContextFree(sys);
  if (map != MAP_FAILED) munmap(map, size);
  if (fd >= 0) close(fd);
  if (!s && rt) releaseRuntime();
  env->ReleaseStringUTFChars(jpath, path);
  env->ReleaseStringUTFChars(jlibdir, libdir);
  return reinterpret_cast<jlong>(s);
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_qnnkit_QnnSession_nativeLastError(JNIEnv* env, jclass) {
  std::lock_guard<std::mutex> lock(gLock);
  return env->NewStringUTF(gLastError.c_str());
}

extern "C" JNIEXPORT jint JNICALL
Java_io_qnnkit_QnnSession_nativeCount(JNIEnv*, jclass, jlong h, jboolean output) {
  return (jint)tensorCount(fromHandle(h), output == JNI_TRUE);
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_qnnkit_QnnSession_nativeName(JNIEnv* env, jclass, jlong h, jboolean output, jint i) {
  return env->NewStringUTF(QNN_TENSOR_GET_NAME(tensorAt(fromHandle(h), output == JNI_TRUE, i)));
}

extern "C" JNIEXPORT jintArray JNICALL
Java_io_qnnkit_QnnSession_nativeDims(JNIEnv* env, jclass, jlong h, jboolean output, jint i) {
  const Qnn_Tensor_t& t = tensorAt(fromHandle(h), output == JNI_TRUE, i);
  uint32_t rank = QNN_TENSOR_GET_RANK(t);
  jintArray dims = env->NewIntArray((jsize)rank);
  std::vector<jint> v(rank);
  for (uint32_t k = 0; k < rank; k++) v[k] = (jint)QNN_TENSOR_GET_DIMENSIONS(t)[k];
  env->SetIntArrayRegion(dims, 0, (jsize)rank, v.data());
  return dims;
}

extern "C" JNIEXPORT jint JNICALL
Java_io_qnnkit_QnnSession_nativeDataType(JNIEnv*, jclass, jlong h, jboolean output, jint i) {
  return (jint)QNN_TENSOR_GET_DATA_TYPE(tensorAt(fromHandle(h), output == JNI_TRUE, i));
}

extern "C" JNIEXPORT jobject JNICALL
Java_io_qnnkit_QnnSession_nativeBuffer(JNIEnv* env, jclass, jlong h, jboolean output, jint i) {
  Qnn_ClientBuffer_t b = QNN_TENSOR_GET_CLIENT_BUF(tensorAt(fromHandle(h), output == JNI_TRUE, i));
  return env->NewDirectByteBuffer(b.data, (jlong)b.dataSize);
}

extern "C" JNIEXPORT jint JNICALL
Java_io_qnnkit_QnnSession_nativeExecute(JNIEnv*, jclass, jlong h) {
  Session* s = fromHandle(h);
  auto& g = (*s->graphs)[0];
  return (jint)gRt->qfp.qnnInterface.graphExecute(g.graph, s->inputs, g.numInputTensors, s->outputs,
                                                  g.numOutputTensors, nullptr, nullptr);
}

extern "C" JNIEXPORT void JNICALL
Java_io_qnnkit_QnnSession_nativeClose(JNIEnv*, jclass, jlong h) {
  std::lock_guard<std::mutex> lock(gLock);
  Session* s = fromHandle(h);
  if (!s) return;
  auto& g = (*s->graphs)[0];
  s->io.tearDownInputAndOutputTensors(s->inputs, s->outputs, g.numInputTensors, g.numOutputTensors);
  if (s->context && gRt) gRt->qfp.qnnInterface.contextFree(s->context, nullptr);
  delete s;
  releaseRuntime();
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_qnnkit_QnnSession_nativeSetBurst(JNIEnv*, jclass, jboolean burst) {
  std::lock_guard<std::mutex> lock(gLock);
  return (gRt && setPower(gRt, burst == JNI_TRUE)) ? JNI_TRUE : JNI_FALSE;
}
