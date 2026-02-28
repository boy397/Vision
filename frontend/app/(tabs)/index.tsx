import React, { useState, useRef, useCallback, useEffect } from "react";
import {
  StyleSheet,
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  SafeAreaView,
  Dimensions,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import { Audio } from "expo-av";
import api, { type ScanResult, type Detection } from "@/services/api";
import scanEvents, { type ScanCommand } from "@/services/scanEvents";

const { width: SCREEN_WIDTH } = Dimensions.get("window");
const CAMERA_HEIGHT = 450;

export default function ScannerScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [mode, setMode] = useState<"medical" | "retail">("medical");
  const [scanState, setScanState] = useState<"idle" | "single" | "continuous">(
    "idle",
  );
  const [statusText, setStatusText] = useState('Say "scan" to capture');
  const cameraRef = useRef<CameraView>(null);
  const isScanningRef = useRef(false);
  const continuousTimerRef = useRef<ReturnType<typeof setInterval> | null>(
    null,
  );
  const soundRef = useRef<Audio.Sound | null>(null);

  // Track which objects have already been analyzed (by class + rough bbox region)
  const analyzedObjectsRef = useRef<Set<string>>(new Set());

  // ── Single scan ──
  const performScan = useCallback(async (skipTracked: boolean = false) => {
    if (!cameraRef.current || isScanningRef.current) return;

    isScanningRef.current = true;
    setScanning(true);
    const start = Date.now();

    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.7,
        base64: false,
      });

      if (!photo?.uri) return;

      // First, get JSON result for detections and bounding boxes
      const data = await api.scanFromUri(photo.uri);
      setResult(data);
      setLatency(Date.now() - start);
      if (data.mode) setMode(data.mode as "medical" | "retail");

      // Object tracking: check if we already analyzed this object
      if (skipTracked && data.detections?.length > 0) {
        const objectKey = data.detections
          .map(
            (d: Detection) =>
              `${d.class_name}_${Math.round(d.bbox[0] / 50)}_${Math.round(d.bbox[1] / 50)}`,
          )
          .sort()
          .join("|");

        if (analyzedObjectsRef.current.has(objectKey)) {
          // Same object(s) in same position — skip TTS
          return;
        }
        analyzedObjectsRef.current.add(objectKey);
      }

      // Now get TTS audio via lightweight /tts endpoint
      if (data.tts_text) {
        try {
          const audioRes = await api.speakText(data.tts_text);
          const contentType = audioRes.headers.get("content-type") || "";
          if (contentType.includes("audio")) {
            // Get audio as blob, write to temp, play
            const blob = await audioRes.blob();
            const reader = new FileReader();
            reader.onloadend = async () => {
              const base64 = (reader.result as string).split(",")[1];
              if (base64) {
                const { sound } = await Audio.Sound.createAsync(
                  { uri: `data:audio/mpeg;base64,${base64}` },
                  { shouldPlay: true },
                );
                // Clean up previous sound
                if (soundRef.current) {
                  await soundRef.current.unloadAsync().catch(() => {});
                }
                soundRef.current = sound;
              }
            };
            reader.readAsDataURL(blob);
          }
        } catch (ttsErr) {
          console.warn("TTS playback error:", ttsErr);
        }
      }
    } catch (err) {
      console.error("Scan error:", err);
      setStatusText("Scan failed — try again");
    } finally {
      setScanning(false);
      isScanningRef.current = false;
    }
  }, []);

  // ── Continuous scanning (every ~750ms ≈ 15-20 frames at 30fps) ──
  const startContinuous = useCallback(() => {
    setScanState("continuous");
    setStatusText('Continuous scanning... Say "scan stop"');
    analyzedObjectsRef.current.clear();

    if (continuousTimerRef.current) clearInterval(continuousTimerRef.current);
    continuousTimerRef.current = setInterval(() => {
      performScan(true); // skip already-tracked objects
    }, 750);
  }, [performScan]);

  const stopContinuous = useCallback(() => {
    if (continuousTimerRef.current) {
      clearInterval(continuousTimerRef.current);
      continuousTimerRef.current = null;
    }
    setScanState("idle");
    setStatusText('Say "scan" to capture');
    analyzedObjectsRef.current.clear();
  }, []);

  // ── Listen for voice-driven scan events ──
  useEffect(() => {
    const unsubscribe = scanEvents.on((command: ScanCommand) => {
      switch (command) {
        case "scan":
          setScanState("single");
          setStatusText("Scanning...");
          performScan(false).then(() => {
            setScanState("idle");
            setStatusText('Say "scan" to capture');
          });
          break;
        case "scan_continue":
          startContinuous();
          break;
        case "scan_stop":
          stopContinuous();
          break;
      }
    });

    return () => {
      unsubscribe();
      if (continuousTimerRef.current) clearInterval(continuousTimerRef.current);
    };
  }, [performScan, startContinuous, stopContinuous]);

  // ── Cleanup audio on unmount ──
  useEffect(() => {
    return () => {
      if (soundRef.current) {
        soundRef.current.unloadAsync().catch(() => {});
      }
    };
  }, []);

  if (!permission) return <View style={styles.container} />;

  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.permissionBox}>
          <Text style={styles.permissionTitle}>📷 Camera Access</Text>
          <Text style={styles.permissionText}>
            The camera is needed to scan medicines and products.
          </Text>
          <TouchableOpacity
            style={styles.permissionBtn}
            onPress={requestPermission}
          >
            <Text style={styles.permissionBtnText}>Grant Access</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      {/* Camera with bounding box overlay */}
      <View style={styles.cameraWrap}>
        <CameraView ref={cameraRef} style={styles.camera} facing="back" />

        {/* Bounding box overlays */}
        {result?.detections?.map((det: Detection, i: number) => {
          const [x1, y1, x2, y2] = det.bbox;
          // Scale bbox coords relative to camera view
          const scaleX = (SCREEN_WIDTH - 24) / 640;
          const scaleY = CAMERA_HEIGHT / 480;
          return (
            <View
              key={i}
              style={[
                styles.bboxOverlay,
                {
                  left: x1 * scaleX,
                  top: y1 * scaleY,
                  width: (x2 - x1) * scaleX,
                  height: (y2 - y1) * scaleY,
                  borderColor: mode === "medical" ? "#6c5ce7" : "#00cec9",
                },
              ]}
            >
              <View
                style={[
                  styles.bboxLabel,
                  {
                    backgroundColor: mode === "medical" ? "#6c5ce7" : "#00cec9",
                  },
                ]}
              >
                <Text style={styles.bboxLabelText}>
                  {det.class_name} {(det.confidence * 100).toFixed(0)}%
                </Text>
              </View>
            </View>
          );
        })}

        {/* Scanning Indicator */}
        {scanning && (
          <View style={styles.scanIndicator}>
            <ActivityIndicator
              size="small"
              color={mode === "medical" ? "#6c5ce7" : "#00cec9"}
            />
          </View>
        )}

        {/* Mode Badge */}
        <View
          style={[
            styles.modeBadge,
            { backgroundColor: mode === "medical" ? "#6c5ce722" : "#00cec922" },
          ]}
        >
          <Text style={styles.modeBadgeText}>
            {mode === "medical" ? "💊 Medical" : "🛒 Retail"}
          </Text>
        </View>

        {/* Scan State Badge */}
        <View
          style={[
            styles.scanStateBadge,
            {
              backgroundColor:
                scanState === "continuous"
                  ? "#00b89422"
                  : scanState === "single"
                    ? "#fdcb6e22"
                    : "#636e7222",
            },
          ]}
        >
          <View
            style={[
              styles.scanStateDot,
              {
                backgroundColor:
                  scanState === "continuous"
                    ? "#00b894"
                    : scanState === "single"
                      ? "#fdcb6e"
                      : "#636e72",
              },
            ]}
          />
          <Text style={styles.scanStateText}>
            {scanState === "continuous"
              ? "CONTINUOUS"
              : scanState === "single"
                ? "SCANNING"
                : "IDLE"}
          </Text>
        </View>

        {/* Bottom status bar */}
        <View style={styles.transcriptOverlay}>
          <Text style={styles.transcriptLabel}>{statusText.toUpperCase()}</Text>
          <Text style={styles.transcriptText} numberOfLines={2}>
            {result?.tts_text || 'Point camera at an object and say "scan"'}
          </Text>
        </View>
      </View>

      {/* Results */}
      <ScrollView
        style={styles.results}
        contentContainerStyle={styles.resultsContent}
      >
        {latency !== null && (
          <View style={styles.latencyBadge}>
            <Text style={styles.latencyText}>{latency}ms</Text>
          </View>
        )}

        {/* Manual scan button (fallback) */}
        <TouchableOpacity
          style={[
            styles.manualScanBtn,
            {
              backgroundColor: mode === "medical" ? "#6c5ce7" : "#00cec9",
              opacity: scanning ? 0.5 : 1,
            },
          ]}
          onPress={() => performScan(false)}
          disabled={scanning}
        >
          <Text style={styles.manualScanBtnText}>
            {scanning ? "⏳ Scanning..." : "📷 Manual Scan"}
          </Text>
        </TouchableOpacity>

        {result ? (
          <>
            {/* Detections */}
            {result.detections.length > 0 && (
              <View style={styles.section}>
                {result.detections.map((det: Detection, i: number) => (
                  <View
                    key={i}
                    style={[
                      styles.detectionItem,
                      {
                        borderLeftColor:
                          mode === "medical" ? "#6c5ce7" : "#00cec9",
                      },
                    ]}
                  >
                    <Text style={styles.detLabel}>{det.class_name}</Text>
                    <Text style={styles.detConf}>
                      {(det.confidence * 100).toFixed(1)}%
                    </Text>
                  </View>
                ))}
              </View>
            )}

            {/* Analysis Detail */}
            {result.analysis &&
              !result.analysis.error &&
              Object.keys(result.analysis).length > 0 && (
                <View style={styles.analysisBox}>
                  <Text style={styles.analysisTitle}>📋 Detailed Analysis</Text>
                  {Object.entries(result.analysis).map(([key, val]) => {
                    if (key === "confidence" || key === "raw_text_detected")
                      return null;
                    if (val && typeof val === "object") {
                      return Object.entries(val as Record<string, any>).map(
                        ([k, v]) =>
                          v ? (
                            <Text
                              key={`${key}.${k}`}
                              style={styles.analysisLine}
                            >
                              <Text style={styles.analysisKey}>{k}: </Text>
                              {String(v)}
                            </Text>
                          ) : null,
                      );
                    }
                    return val ? (
                      <Text key={key} style={styles.analysisLine}>
                        <Text style={styles.analysisKey}>{key}: </Text>
                        {String(val)}
                      </Text>
                    ) : null;
                  })}
                </View>
              )}
          </>
        ) : (
          <Text style={styles.placeholder}>
            Point camera at a medicine or product, then say "scan"
          </Text>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0a0a0f" },
  // Permission
  permissionBox: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 32,
  },
  permissionTitle: {
    fontSize: 24,
    fontWeight: "700",
    color: "#f0f0f5",
    marginBottom: 12,
  },
  permissionText: {
    fontSize: 16,
    color: "#8888a0",
    textAlign: "center",
    marginBottom: 24,
  },
  permissionBtn: {
    backgroundColor: "#6c5ce7",
    paddingHorizontal: 32,
    paddingVertical: 14,
    borderRadius: 12,
  },
  permissionBtnText: { color: "#fff", fontSize: 16, fontWeight: "600" },
  // Camera
  cameraWrap: {
    height: CAMERA_HEIGHT,
    marginHorizontal: 12,
    marginTop: 8,
    borderRadius: 24,
    overflow: "hidden",
    position: "relative",
    elevation: 10,
  },
  camera: { flex: 1 },
  scanIndicator: {
    position: "absolute",
    top: 12,
    left: 12,
    backgroundColor: "#00000088",
    padding: 8,
    borderRadius: 20,
  },
  modeBadge: {
    position: "absolute",
    top: 12,
    right: 12,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  modeBadgeText: { color: "#f0f0f5", fontSize: 13, fontWeight: "600" },
  // Scan State
  scanStateBadge: {
    position: "absolute",
    top: 12,
    left: 50,
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 16,
  },
  scanStateDot: { width: 8, height: 8, borderRadius: 4 },
  scanStateText: {
    color: "#f0f0f5",
    fontSize: 10,
    fontWeight: "700",
    letterSpacing: 1,
  },
  // Bounding boxes
  bboxOverlay: {
    position: "absolute",
    borderWidth: 2,
    borderRadius: 4,
  },
  bboxLabel: {
    position: "absolute",
    top: -18,
    left: -2,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  bboxLabelText: {
    color: "#fff",
    fontSize: 10,
    fontWeight: "700",
  },
  // Transcript
  transcriptOverlay: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: "rgba(10, 10, 15, 0.85)",
    padding: 16,
    borderTopWidth: 1,
    borderTopColor: "#ffffff22",
  },
  transcriptLabel: {
    color: "#00cec9",
    fontSize: 10,
    fontWeight: "800",
    marginBottom: 4,
    letterSpacing: 1,
  },
  transcriptText: {
    color: "#f0f0f5",
    fontSize: 15,
    fontWeight: "500",
    lineHeight: 20,
  },
  // Results
  results: { flex: 1, marginTop: 16 },
  resultsContent: { paddingHorizontal: 12, paddingBottom: 20 },
  latencyBadge: {
    alignSelf: "flex-end",
    backgroundColor: "#12121a",
    paddingHorizontal: 10,
    paddingVertical: 3,
    borderRadius: 8,
    marginBottom: 12,
  },
  latencyText: {
    color: "#55556a",
    fontSize: 12,
    fontVariant: ["tabular-nums"],
  },
  placeholder: {
    color: "#55556a",
    textAlign: "center",
    marginTop: 32,
    fontSize: 14,
  },
  section: { marginBottom: 16 },
  // Manual scan
  manualScanBtn: {
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: "center",
    marginBottom: 16,
  },
  manualScanBtnText: { color: "#fff", fontSize: 16, fontWeight: "600" },
  // Detection Item
  detectionItem: {
    backgroundColor: "#12121a",
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
    borderLeftWidth: 3,
    flexDirection: "row",
    justifyContent: "space-between",
  },
  detLabel: { color: "#f0f0f5", fontWeight: "600", fontSize: 14 },
  detConf: { color: "#00b894", fontSize: 13, fontVariant: ["tabular-nums"] },
  // Analysis
  analysisBox: {
    backgroundColor: "#12121a",
    borderRadius: 12,
    padding: 16,
    marginTop: 4,
  },
  analysisTitle: {
    color: "#6c5ce7",
    fontWeight: "700",
    fontSize: 14,
    marginBottom: 12,
  },
  analysisLine: {
    color: "#8888a0",
    fontSize: 13,
    lineHeight: 20,
    marginBottom: 4,
  },
  analysisKey: { color: "#f0f0f5", fontWeight: "600" },
});
