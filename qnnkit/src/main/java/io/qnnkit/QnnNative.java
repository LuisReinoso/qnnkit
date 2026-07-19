package io.qnnkit;

/** Thin JNI layer over the QNN HTP (Hexagon NPU) runtime. Package-private:
 *  apps use {@link Npu} / {@link QnnModel}. */
final class QnnNative {
  static { System.loadLibrary("qnnkit"); }

  private QnnNative() {}

  /** Device-create diagnostic. socModel 0 = let the HTP detect the attached SoC. */
  static native String probe(String libDir, int socModel);

  /**
   * One forward pass on the NPU.
   *
   * @param libDir    app native lib dir (getApplicationInfo().nativeLibraryDir)
   * @param modelPath a QAIRT DLC path (isDlc=true) or a QNN model .so path (isDlc=false)
   * @param isDlc     true to load via libQnnModelDlc.so, false for a .so graph
   * @param backend   path to libQnnHtp.so
   * @param cachePath where to read/write the finalized context binary
   * @param graphName the graph name inside the model (also keys the FP16 config)
   * @param socModel  QNN SoC id, or 0 to auto-detect
   * @return first output tensor's floats, or null on failure (logcat tag "qnnkit")
   */
  static native float[] run(String libDir, String modelPath, boolean isDlc, String backend,
                            String cachePath, String graphName, int socModel, float[] input);
}
