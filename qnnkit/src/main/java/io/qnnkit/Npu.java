package io.qnnkit;

import android.content.Context;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Entry point for running models on the Qualcomm Hexagon NPU, offline, inside your
 * app process. No server, no NNAPI, no extra runtime: just the bundled QNN libs.
 *
 * <pre>{@code
 * if (Npu.isAvailable(ctx)) {
 *   QnnModel det = Npu.loadDlc(ctx, "detector.dlc", "detector_qairt"); // from assets/
 *   QnnModel rec = Npu.loadSo(ctx, "librecognizer.so", "recognizer");  // from jniLibs/
 *   float[] heat = det.run(input);
 * }
 * }</pre>
 *
 * The SoC is detected at runtime ({@link SocInfo}); on unknown Snapdragons the HTP
 * auto-detects the chip. The context binary is cached per model (keyed by the model
 * file size), so swapping a model file re-finalizes automatically.
 */
public final class Npu {
  private Npu() {}

  /** Detected SoC (chip model, Qualcomm-or-not, QNN SoC id). */
  public static SocInfo soc() { return SocInfo.detect(); }

  /** True when this is a Qualcomm device and the QNN HTP runtime lib is bundled. */
  public static boolean isAvailable(Context c) {
    return SocInfo.detect().qualcomm && new File(libDir(c), "libQnnHtp.so").exists();
  }

  /** Device-create diagnostic string (also logged under logcat tag "qnnkit"). */
  public static String probe(Context c) {
    return QnnNative.probe(libDir(c), SocInfo.detect().qnnSocModel);
  }

  /**
   * Load a QAIRT DLC (qairt-converter output). {@code dlc} is either an assets/ file
   * name (copied to the cache dir) or an absolute file path. {@code graphName} is the
   * graph inside the DLC (visible in the "finalize=... graph=..." log line).
   */
  public static QnnModel loadDlc(Context c, String dlc, String graphName) {
    String path = resolve(c, dlc);
    return new QnnModel(libDir(c), path, true, backend(c),
        cacheFor(c, graphName, path), graphName, SocInfo.detect().qnnSocModel);
  }

  /**
   * Load a QNN model .so (qnn-onnx-converter + qnn-model-lib-generator output) that
   * ships in the app's jniLibs, e.g. {@code "librecognizer.so"}.
   */
  public static QnnModel loadSo(Context c, String soFileName, String graphName) {
    String path = new File(libDir(c), soFileName).getAbsolutePath();
    return new QnnModel(libDir(c), path, false, backend(c),
        cacheFor(c, graphName, path), graphName, SocInfo.detect().qnnSocModel);
  }

  // --- helpers -------------------------------------------------------------

  private static String libDir(Context c) {
    return c.getApplicationInfo().nativeLibraryDir;
  }

  private static String backend(Context c) {
    return new File(libDir(c), "libQnnHtp.so").getAbsolutePath();
  }

  /** Cache path keyed by graph + model file size, so a changed model re-finalizes. */
  private static String cacheFor(Context c, String graphName, String modelPath) {
    long fp = new File(modelPath).length();
    return new File(c.getCacheDir(), "qnnkit_" + safe(graphName) + "_" + fp + ".bin")
        .getAbsolutePath();
  }

  /** Absolute path if {@code name} points at a file; otherwise copy the asset to
   *  the cache dir and return that (composeGraphsFromDlc needs a real file path). */
  private static String resolve(Context c, String name) {
    File direct = new File(name);
    if (direct.isAbsolute() && direct.exists()) return direct.getAbsolutePath();
    File dest = new File(c.getCacheDir(), "qnnkit_asset_" + safe(name));
    try (InputStream in = c.getAssets().open(name);
         OutputStream out = new FileOutputStream(dest)) {
      byte[] b = new byte[1 << 16]; int n;
      while ((n = in.read(b)) > 0) out.write(b, 0, n);
    } catch (Exception e) {
      // If it wasn't an asset, fall back to treating it as a path.
      return direct.getAbsolutePath();
    }
    return dest.getAbsolutePath();
  }

  private static String safe(String s) {
    return s == null ? "x" : s.replaceAll("[^A-Za-z0-9_.-]", "_");
  }
}
