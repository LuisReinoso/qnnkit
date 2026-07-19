package io.qnnkit;

/**
 * A model loaded onto the NPU. Obtain one via {@link Npu#loadDlc} or
 * {@link Npu#loadSo}, then call {@link #run} as many times as you like.
 *
 * <p>Each {@code run} composes + finalizes on the first call (result cached to a
 * context binary), so subsequent calls skip the multi-second finalize. QNN handles
 * are created and freed per call, so there is no leak and nothing to close.
 */
public final class QnnModel {
  private final String libDir;
  private final String modelPath;
  private final boolean isDlc;
  private final String backend;
  private final String cachePath;
  private final String graphName;
  private final int socModel;

  QnnModel(String libDir, String modelPath, boolean isDlc, String backend,
           String cachePath, String graphName, int socModel) {
    this.libDir = libDir;
    this.modelPath = modelPath;
    this.isDlc = isDlc;
    this.backend = backend;
    this.cachePath = cachePath;
    this.graphName = graphName;
    this.socModel = socModel;
  }

  /**
   * Runs one forward pass. The input is copied into the graph's first input tensor
   * (row-major, in the layout the graph expects); the first output tensor's floats
   * are returned. Returns null if inference failed (see logcat tag "qnnkit").
   */
  public float[] run(float[] input) {
    return QnnNative.run(libDir, modelPath, isDlc, backend, cachePath, graphName, socModel, input);
  }

  public String graphName() { return graphName; }
}
