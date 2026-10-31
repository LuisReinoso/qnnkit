package io.qnnkit;

import java.io.Closeable;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * A prebuilt QNN context binary kept loaded on the NPU. Unlike {@link QnnModel}, which
 * composes and frees everything per call, a session pays the load once and then each
 * {@link #execute()} only runs the graph, so it suits models called many times in a row.
 *
 * <p>Context binaries are compiled offline for the target SoC (qnn-context-binary-generator)
 * with the same QAIRT version as the runtime bundled here. Graphs can have any number of
 * named inputs and outputs: write inputs into {@link #input(String)}, call {@link #execute()},
 * read {@link #output(String)}. Both are direct buffers over the tensors' own memory, in
 * native byte order. A session is not thread safe; serialize calls on it.
 *
 * <pre>{@code
 * try (QnnSession enc = Npu.openContext(ctx, new File(dir, "encoder.bin"))) {
 *   enc.input("x").asIntBuffer().put(ids);
 *   if (enc.execute()) enc.output("m_p").asFloatBuffer().get(out);
 * }
 * }</pre>
 */
public final class QnnSession implements Closeable {
  /** QNN data type codes (Qnn_DataType_t) for the common cases. */
  public static final int FLOAT_32 = 0x0232, INT_32 = 0x0032, FLOAT_16 = 0x0216, UINT_8 = 0x0108;

  static { System.loadLibrary("qnnkit"); }

  private long handle;
  private final String[] inNames, outNames;
  private final ByteBuffer[] inBufs, outBufs;

  private QnnSession(long handle) {
    this.handle = handle;
    inNames = names(false);
    outNames = names(true);
    inBufs = buffers(false, inNames.length);
    outBufs = buffers(true, outNames.length);
  }

  /** Loads a context binary; null if it cannot be loaded (see logcat tag "qnnkit"). */
  static QnnSession open(String libDir, File contextBinary, int socModel) {
    long h = nativeOpen(libDir, contextBinary.getAbsolutePath(), socModel);
    return h == 0 ? null : new QnnSession(h);
  }

  public String[] inputNames() { return inNames.clone(); }

  public String[] outputNames() { return outNames.clone(); }

  public int[] inputShape(String name) { return nativeDims(handle, false, index(inNames, name)); }

  public int[] outputShape(String name) { return nativeDims(handle, true, index(outNames, name)); }

  public int inputDataType(String name) { return nativeDataType(handle, false, index(inNames, name)); }

  public int outputDataType(String name) { return nativeDataType(handle, true, index(outNames, name)); }

  /** The input tensor's memory, positioned at 0. */
  public ByteBuffer input(String name) {
    ByteBuffer b = inBufs[index(inNames, name)];
    b.clear();
    return b;
  }

  /** The output tensor's memory, positioned at 0; valid after a successful {@link #execute()}. */
  public ByteBuffer output(String name) {
    ByteBuffer b = outBufs[index(outNames, name)];
    b.clear();
    return b;
  }

  /** Runs the graph on the current inputs. False on failure (error code in logcat). */
  public boolean execute() {
    check();
    int err = nativeExecute(handle);
    if (err != 0) android.util.Log.e("qnnkit", "graphExecute failed: " + err);
    return err == 0;
  }

  @Override public void close() {
    if (handle != 0) {
      nativeClose(handle);
      handle = 0;
    }
  }

  /** Why the last open on this process failed (empty if it did not), e.g. the QNN step and
   *  error code. Also in logcat under "qnnkit". */
  public static String lastError() { return nativeLastError(); }

  /**
   * Votes the NPU clocks to burst (true) or back to DCVS (false) for every open session.
   * Hold burst only while work is queued; it keeps the HTP at max voltage.
   */
  public static boolean setBurst(boolean burst) { return nativeSetBurst(burst); }

  private String[] names(boolean output) {
    String[] n = new String[nativeCount(handle, output)];
    for (int i = 0; i < n.length; i++) n[i] = nativeName(handle, output, i);
    return n;
  }

  private ByteBuffer[] buffers(boolean output, int count) {
    ByteBuffer[] b = new ByteBuffer[count];
    for (int i = 0; i < count; i++) b[i] = nativeBuffer(handle, output, i).order(ByteOrder.nativeOrder());
    return b;
  }

  private int index(String[] names, String name) {
    check();
    for (int i = 0; i < names.length; i++) if (names[i].equals(name)) return i;
    throw new IllegalArgumentException("no tensor '" + name + "' in " + Arrays.toString(names));
  }

  private void check() {
    if (handle == 0) throw new IllegalStateException("session closed");
  }

  private static native long nativeOpen(String libDir, String path, int socModel);
  private static native String nativeLastError();
  private static native int nativeCount(long h, boolean output);
  private static native String nativeName(long h, boolean output, int i);
  private static native int[] nativeDims(long h, boolean output, int i);
  private static native int nativeDataType(long h, boolean output, int i);
  private static native ByteBuffer nativeBuffer(long h, boolean output, int i);
  private static native int nativeExecute(long h);
  private static native void nativeClose(long h);
  private static native boolean nativeSetBurst(boolean burst);
}
