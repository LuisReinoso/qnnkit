package io.qnnkit;

import android.os.Build;

/**
 * Runtime SoC detection. The NPU path only exists on Qualcomm Snapdragon (Hexagon);
 * {@link #qnnSocModel} is the QNN SoC id to pass at device-create time, or 0 to let
 * the HTP backend detect the attached chip on-device.
 */
public final class SocInfo {
  /** e.g. "SM8450" on API 31+, else null. */
  public final String model;
  /** e.g. "QTI" / "Qualcomm", else null. */
  public final String manufacturer;
  /** True when this looks like a Qualcomm Snapdragon (has a Hexagon NPU). */
  public final boolean qualcomm;
  /** QNN SoC id, or 0 = HTP auto-detect. */
  public final int qnnSocModel;

  private SocInfo(String model, String manufacturer, boolean qualcomm, int qnnSocModel) {
    this.model = model;
    this.manufacturer = manufacturer;
    this.qualcomm = qualcomm;
    this.qnnSocModel = qnnSocModel;
  }

  public static SocInfo detect() {
    String model = null, mfr = null;
    if (Build.VERSION.SDK_INT >= 31) {
      model = Build.SOC_MODEL;
      mfr = Build.SOC_MANUFACTURER;
    }
    boolean q = contains(mfr, "qualcomm") || contains(mfr, "qti")
        || startsWith(Build.HARDWARE, "qcom") || contains(Build.BOARD, "msm")
        || contains(model, "sm8") || contains(model, "sm7") || contains(model, "sm6");
    return new SocInfo(model, mfr, q, mapSoc(model));
  }

  /**
   * Maps a Build.SOC_MODEL string to a QNN SoC id. Only SM8450 (Snapdragon 8 Gen 1,
   * Galaxy S22, Hexagon v73) is verified; other chips fall back to 0 (auto-detect).
   * Extend from the SocModel enum in the QNN SDK's QnnHtpDevice.h.
   */
  private static int mapSoc(String m) {
    if (m == null) return 0;
    switch (m.toUpperCase()) {
      case "SM8450": return 36;   // verified: Snapdragon 8 Gen 1, Hexagon v73
      default:       return 0;    // let the HTP detect the online device
    }
  }

  private static boolean contains(String s, String needle) {
    return s != null && s.toLowerCase().contains(needle);
  }

  private static boolean startsWith(String s, String prefix) {
    return s != null && s.toLowerCase().startsWith(prefix);
  }

  @Override public String toString() {
    return "SocInfo{model=" + model + ", mfr=" + manufacturer
        + ", qualcomm=" + qualcomm + ", qnnSocModel=" + qnnSocModel + "}";
  }
}
