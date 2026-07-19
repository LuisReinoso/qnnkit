// qnnkit - reusable raw-QNN inference for Android (Qualcomm Hexagon NPU).
//
// JNI backing io.qnnkit.QnnNative. Two entry points:
//   probe(libDir, socModel)  -> a human-readable device-create diagnostic
//   run(libDir, modelPath, isDlc, backend, cachePath, graphName, socModel, in) -> out
//
// One code path serves both a QNN model .so (qnn-onnx-converter output) and a
// QAIRT DLC (qairt-converter output, loaded via libQnnModelDlc.so). Float graphs
// run on the HTP in FP16; the finalized graph is cached as a context binary so
// later runs skip the multi-second finalize. socModel == 0 lets the HTP detect
// the attached SoC (on-device); a non-zero value forces a specific SoC.

#include <jni.h>
#include <android/log.h>
#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <string>
#include <vector>
#include <chrono>

#include "QNN/QnnInterface.h"
#include "QNN/QnnBackend.h"
#include "QNN/QnnContext.h"
#include "QNN/QnnDevice.h"
#include "QNN/QnnGraph.h"
#include "QNN/QnnLog.h"
#include "QNN/HTP/QnnHtpDevice.h"
#include "QNN/HTP/QnnHtpGraph.h"
#include "QNN/System/QnnSystemContext.h"

#include "SampleApp.hpp"
#include "Utils/DynamicLoadUtil.hpp"
#include "Utils/IOTensor.hpp"
#include "WrapperUtils/QnnWrapperUtils.hpp"
#include "QnnTypeMacros.hpp"

using namespace qnn::tools;

#define TAG "qnnkit"

static void qnnLog(const char* fmt, QnnLog_Level_t lvl, uint64_t ts, va_list ap) {
  char b[1024]; vsnprintf(b, sizeof(b), fmt, ap);
  __android_log_print(ANDROID_LOG_INFO, "QnnLib", "%s", b);
}

typedef Qnn_ErrorHandle_t (*GetProvidersFn)(const QnnInterface_t***, uint32_t*);

// Signature of QnnModel_composeGraphsFromDlc (from libQnnModelDlc.so): dlcPath is
// arg 5, right after the graph-config array.
typedef qnn_wrapper_api::ModelError_t (*ComposeFromDlcFn)(
    Qnn_BackendHandle_t, QNN_INTERFACE_VER_TYPE, Qnn_ContextHandle_t,
    const qnn_wrapper_api::GraphConfigInfo_t**, const char*, const uint32_t,
    qnn_wrapper_api::GraphInfo_t***, uint32_t*, bool, QnnLog_Callback_t, QnnLog_Level_t);

// Frees the per-call QNN handles on EVERY return path. Without this the context
// (which holds the graph's DSP memory) leaks each call, so the 2nd big-model run
// fails to allocate on the NPU -> graphExecute returns 6001 (MEM_ALLOC). Holds
// pointers so the handles can be created after the guard is armed.
namespace {
struct QnnScope {
  QNN_INTERFACE_VER_TYPE* itf;
  Qnn_ContextHandle_t* ctx;
  Qnn_DeviceHandle_t* dev;
  Qnn_BackendHandle_t* bk;
  Qnn_LogHandle_t* logH;
  ~QnnScope() {
    if (ctx && *ctx && itf->contextFree)  itf->contextFree(*ctx, nullptr);
    if (dev && *dev && itf->deviceFree)    itf->deviceFree(*dev);
    if (bk  && *bk  && itf->backendFree)   itf->backendFree(*bk);
    if (logH && *logH && itf->logFree)     itf->logFree(*logH);
  }
};
}

// Points the CDSP at the app's native lib dir (where the Hexagon skels live) and
// preloads the vendor FastRPC lib by full path with RTLD_GLOBAL - the app linker
// namespace won't resolve it by soname otherwise.
static void prepareDsp(const char* libdir) {
  setenv("ADSP_LIBRARY_PATH", libdir, 1);
  dlopen("/vendor/lib64/libcdsprpc.so", RTLD_NOW | RTLD_GLOBAL);
}

// Builds the device config: always request an unsigned process domain (no OEM
// signing needed); add a fixed SoC only when socModel != 0.
static void deviceConfigs(int socModel,
                          QnnHtpDevice_CustomConfig_t& soc, QnnHtpDevice_CustomConfig_t& pd,
                          QnnDevice_Config_t& cSoc, QnnDevice_Config_t& cPd,
                          std::vector<const QnnDevice_Config_t*>& out) {
  pd.option = QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD;
  pd.useSignedProcessDomain.deviceId = 0;
  pd.useSignedProcessDomain.useSignedProcessDomain = false;
  cPd.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM; cPd.customConfig = &pd;
  if (socModel > 0) {
    soc.option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC; soc.socModel = (uint32_t)socModel;
    cSoc.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM; cSoc.customConfig = &soc;
    out.push_back(&cSoc);
  }
  out.push_back(&cPd);
  out.push_back(nullptr);
}

// ---------------------------------------------------------------------------
// probe: prove the HTP/NPU device can be created inside the app process.
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jstring JNICALL
Java_io_qnnkit_QnnNative_probe(JNIEnv* env, jclass, jstring jlibdir, jint jsoc) {
  const char* libdir = env->GetStringUTFChars(jlibdir, nullptr);
  int socModel = (int)jsoc;
  std::string r;

  prepareDsp(libdir);
  r += std::string("ADSP_LIBRARY_PATH=") + libdir + "\n";
  const char* cdspPaths[] = {
    "/vendor/lib64/libcdsprpc.so", "/system/lib64/libcdsprpc.so", "/vendor/lib64/libadsprpc.so" };
  for (const char* p : cdspPaths) {
    void* c = dlopen(p, RTLD_NOW | RTLD_GLOBAL);
    r += std::string("preload ") + p + (c ? " OK\n" : (std::string(" FAIL: ") + dlerror() + "\n"));
  }

  void* h = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
  if (!h) { r += std::string("dlopen libQnnHtp.so FAILED: ") + dlerror() + "\n";
            env->ReleaseStringUTFChars(jlibdir, libdir); return env->NewStringUTF(r.c_str()); }
  r += "dlopen libQnnHtp.so OK\n";

  auto getProviders = (GetProvidersFn) dlsym(h, "QnnInterface_getProviders");
  if (!getProviders) { r += "dlsym getProviders FAILED\n";
                       env->ReleaseStringUTFChars(jlibdir, libdir); return env->NewStringUTF(r.c_str()); }

  const QnnInterface_t** providers = nullptr; uint32_t num = 0;
  Qnn_ErrorHandle_t err = getProviders(&providers, &num);
  r += "getProviders err=" + std::to_string((long long)err) + " num=" + std::to_string(num) + "\n";
  if (err != QNN_SUCCESS || num == 0 || !providers) {
    env->ReleaseStringUTFChars(jlibdir, libdir); return env->NewStringUTF(r.c_str()); }

  QNN_INTERFACE_VER_TYPE iface = providers[0]->QNN_INTERFACE_VER_NAME;
  Qnn_LogHandle_t logH = nullptr; iface.logCreate(qnnLog, QNN_LOG_LEVEL_INFO, &logH);
  Qnn_BackendHandle_t beH = nullptr;
  err = iface.backendCreate(logH, nullptr, &beH);
  r += "backendCreate err=" + std::to_string((long long)err) + "\n";
  if (err != QNN_SUCCESS) { env->ReleaseStringUTFChars(jlibdir, libdir); return env->NewStringUTF(r.c_str()); }

  QnnHtpDevice_CustomConfig_t soc{}, pd{}; QnnDevice_Config_t cSoc{}, cPd{};
  std::vector<const QnnDevice_Config_t*> cfgs;
  deviceConfigs(socModel, soc, pd, cSoc, cPd, cfgs);
  r += std::string("socModel=") + std::to_string(socModel) + (socModel > 0 ? "\n" : " (auto)\n");

  Qnn_DeviceHandle_t devH = nullptr;
  err = iface.deviceCreate(logH, cfgs.data(), &devH);
  r += "deviceCreate err=" + std::to_string((long long)err) +
       (err == QNN_SUCCESS ? "  <<< NPU DEVICE OK" : "  <<< DEVICE FAIL") + "\n";

  if (err == QNN_SUCCESS && iface.deviceFree) iface.deviceFree(devH);
  if (iface.backendFree) iface.backendFree(beH);
  if (iface.logFree) iface.logFree(logH);
  env->ReleaseStringUTFChars(jlibdir, libdir);
  __android_log_print(ANDROID_LOG_INFO, TAG, "%s", r.c_str());
  return env->NewStringUTF(r.c_str());
}

// ---------------------------------------------------------------------------
// run: one forward pass on the NPU. Loads the model (.so graph or DLC), finalizes
// in FP16 (context-binary cached), executes, returns the first output's floats.
// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jfloatArray JNICALL
Java_io_qnnkit_QnnNative_run(JNIEnv* env, jclass,
    jstring jlibdir, jstring jmodel, jboolean jIsDlc, jstring jbackend,
    jstring jcache, jstring jgraph, jint jsoc, jfloatArray jinput) {
  const char* libdir  = env->GetStringUTFChars(jlibdir, nullptr);
  const char* modelPath = env->GetStringUTFChars(jmodel, nullptr);
  const char* backend = env->GetStringUTFChars(jbackend, nullptr);
  const char* cachePath = env->GetStringUTFChars(jcache, nullptr);
  const char* graphName = env->GetStringUTFChars(jgraph, nullptr);
  bool isDlc = (jIsDlc == JNI_TRUE);
  int socModel = (int)jsoc;

  prepareDsp(libdir);

  // The lib to hand to getQnnFunctionPointers: for a DLC it's the shared DLC
  // loader (dlcPath is passed to composeFromDlc); for a .so it's the model itself.
  std::string modelLib = isDlc ? (std::string(libdir) + "/libQnnModelDlc.so") : std::string(modelPath);
  sample_app::QnnFunctionPointers qfp{};
  void* beH = nullptr; void* modelH = nullptr;
  if (dynamicloadutil::getQnnFunctionPointers(backend, modelLib, &qfp, &beH, true, &modelH)
      != dynamicloadutil::StatusCode::SUCCESS) {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "getFnPtrs fail"); return nullptr; }
  dynamicloadutil::getQnnSystemFunctionPointers(std::string(libdir) + "/libQnnSystem.so", &qfp);
  auto itf = qfp.qnnInterface;

  ComposeFromDlcFn composeDlc = nullptr;
  if (isDlc) {
    composeDlc = (ComposeFromDlcFn) dlsym(modelH, "QnnModel_composeGraphsFromDlc");
    if (!composeDlc) { __android_log_print(ANDROID_LOG_ERROR, TAG, "no composeGraphsFromDlc"); return nullptr; }
  }

  Qnn_LogHandle_t logH = nullptr;
  Qnn_BackendHandle_t bk = nullptr;
  Qnn_DeviceHandle_t dev = nullptr;
  Qnn_ContextHandle_t ctx = nullptr;
  QnnScope _scope{&itf, &ctx, &dev, &bk, &logH};   // frees these on every return path

  itf.logCreate(qnnLog, QNN_LOG_LEVEL_ERROR, &logH);
  if (itf.backendCreate(logH, nullptr, &bk) != QNN_SUCCESS) return nullptr;

  QnnHtpDevice_CustomConfig_t soc{}, pd{}; QnnDevice_Config_t cSoc{}, cPd{};
  std::vector<const QnnDevice_Config_t*> cfgs;
  deviceConfigs(socModel, soc, pd, cSoc, cPd, cfgs);
  if (itf.deviceCreate(logH, cfgs.data(), &dev) != QNN_SUCCESS) return nullptr;

  qnn_wrapper_api::GraphInfo_t** graphs = nullptr; uint32_t num = 0;
  bool cached = false; Qnn_ErrorHandle_t e = QNN_SUCCESS;

  // Fast path: load the finalized graph from the cached context binary.
  FILE* cf = fopen(cachePath, "rb");
  if (cf) {
    fseek(cf, 0, SEEK_END); long sz = ftell(cf); fseek(cf, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz > 0 ? sz : 0);
    bool ok = sz > 0 && fread(buf.data(), 1, sz, cf) == (size_t)sz; fclose(cf);
    if (ok && qfp.qnnSystemInterface.systemContextCreate) {
      QnnSystemContext_Handle_t sysCtx = nullptr;
      if (qfp.qnnSystemInterface.systemContextCreate(&sysCtx) == QNN_SUCCESS) {
        const QnnSystemContext_BinaryInfo_t* bi = nullptr; Qnn_ContextBinarySize_t bsz = 0;
        if (qfp.qnnSystemInterface.systemContextGetBinaryInfo(sysCtx, buf.data(), buf.size(), &bi, &bsz) == QNN_SUCCESS &&
            sample_app::copyMetadataToGraphsInfo(bi, graphs, num) &&
            itf.contextCreateFromBinary(bk, dev, nullptr, buf.data(), buf.size(), &ctx, nullptr) == QNN_SUCCESS) {
          for (uint32_t i = 0; i < num; i++) itf.graphRetrieve(ctx, (*graphs)[i].graphName, &(*graphs)[i].graph);
          cached = true;
        }
        qfp.qnnSystemInterface.systemContextFree(sysCtx);
      }
    }
  }

  // Slow path: compose + finalize the graph, then persist the context binary.
  if (!cached) {
    if (itf.contextCreate(bk, dev, nullptr, &ctx) != QNN_SUCCESS) return nullptr;
    QnnHtpGraph_CustomConfig_t hp{}; hp.option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION; hp.precision = QNN_PRECISION_FLOAT16;
    QnnGraph_Config_t g0{}; g0.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM; g0.customConfig = &hp;
    QnnHtpGraph_CustomConfig_t ho{}; ho.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
    ho.optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG; ho.optimizationOption.floatValue = 3;
    QnnGraph_Config_t g1{}; g1.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM; g1.customConfig = &ho;
    const QnnGraph_Config_t* gc[] = { &g0, &g1, nullptr };
    qnn_wrapper_api::GraphConfigInfo_t gci{}; gci.graphName = (char*)graphName; gci.graphConfigs = gc;
    const qnn_wrapper_api::GraphConfigInfo_t* gcis[] = { &gci, nullptr };

    qnn_wrapper_api::ModelError_t me;
    if (isDlc)
      me = composeDlc(bk, itf, ctx, gcis, modelPath, 1, &graphs, &num, false, qnnLog, QNN_LOG_LEVEL_ERROR);
    else
      me = qfp.composeGraphsFnHandle(bk, itf, ctx, gcis, 1, &graphs, &num, false, qnnLog, QNN_LOG_LEVEL_ERROR);
    if (me != qnn_wrapper_api::MODEL_NO_ERROR || num == 0) return nullptr;

    e = itf.graphFinalize((*graphs)[0].graph, nullptr, nullptr);
    __android_log_print(ANDROID_LOG_INFO, TAG, "finalize=%lld graph=%s", (long long)e, (*graphs)[0].graphName);
    if (e != QNN_SUCCESS) return nullptr;

    Qnn_ContextBinarySize_t req = 0;
    if (itf.contextGetBinarySize && itf.contextGetBinarySize(ctx, &req) == QNN_SUCCESS && req > 0) {
      std::vector<uint8_t> ob(req); Qnn_ContextBinarySize_t wr = 0;
      if (itf.contextGetBinary(ctx, ob.data(), req, &wr) == QNN_SUCCESS) {
        FILE* wf = fopen(cachePath, "wb"); if (wf) { fwrite(ob.data(), 1, wr, wf); fclose(wf); }
      }
    }
  }

  qnn_wrapper_api::GraphInfo_t& g = (*graphs)[0];
  iotensor::IOTensor io;
  Qnn_Tensor_t* inputs = nullptr; Qnn_Tensor_t* outputs = nullptr;
  if (io.setupInputAndOutputTensors(&inputs, &outputs, g) != iotensor::StatusCode::SUCCESS) return nullptr;

  jsize inLen = env->GetArrayLength(jinput);
  jfloat* inData = env->GetFloatArrayElements(jinput, nullptr);
  { auto b = QNN_TENSOR_GET_CLIENT_BUF(&inputs[0]); size_t cap = b.dataSize / sizeof(float);
    memcpy(b.data, inData, (((size_t)inLen < cap) ? (size_t)inLen : cap) * sizeof(float)); }
  env->ReleaseFloatArrayElements(jinput, inData, JNI_ABORT);

  auto a = std::chrono::steady_clock::now();
  e = itf.graphExecute(g.graph, inputs, g.numInputTensors, outputs, g.numOutputTensors, nullptr, nullptr);
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
  __android_log_print(ANDROID_LOG_INFO, TAG, "run[%s] cached=%d execute=%lld (%.1f ms)",
                      graphName, cached ? 1 : 0, (long long)e, ms);

  jfloatArray result = nullptr;
  if (e == QNN_SUCCESS) {
    auto ob = QNN_TENSOR_GET_CLIENT_BUF(&outputs[0]);
    size_t outN = ob.dataSize / sizeof(float);
    result = env->NewFloatArray(outN);
    env->SetFloatArrayRegion(result, 0, outN, (float*)ob.data);
  }
  io.tearDownInputAndOutputTensors(inputs, outputs, g.numInputTensors, g.numOutputTensors);
  return result;
}
