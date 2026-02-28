import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  SafeAreaView,
  ActivityIndicator,
} from "react-native";
import { Audio } from "expo-av";
import api, { type VoiceResult, VOICE_WS_URL } from "@/services/api";
import scanEvents from "@/services/scanEvents";

// Reconnect config
const RECONNECT_BASE_DELAY_MS = 1000;
const RECONNECT_MAX_DELAY_MS = 15000;
const PING_INTERVAL_MS = 15000; // Keepalive ping every 15s

// PCM recording config: 16kHz, 16-bit, mono
const RECORDING_OPTIONS: Audio.RecordingOptions = {
  isMeteringEnabled: false,
  android: {
    extension: ".wav",
    outputFormat: 4, // THREE_GPP
    audioEncoder: 1, // AMR_NB
    sampleRate: 16000,
    numberOfChannels: 1,
    bitRate: 256000,
  },
  ios: {
    extension: ".wav",
    outputFormat: "linearPCM" as any,
    audioQuality: 96, // MEDIUM
    sampleRate: 16000,
    numberOfChannels: 1,
    bitRate: 256000,
    linearPCMBitDepth: 16,
    linearPCMIsBigEndian: false,
    linearPCMIsFloat: false,
  },
  web: {
    mimeType: "audio/webm",
    bitsPerSecond: 128000,
  },
};

export default function VoiceScreen() {
  const [listening, setListening] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [connected, setConnected] = useState(false);
  const [history, setHistory] = useState<
    Array<{ type: "user" | "system" | "status"; text: string }>
  >([]);
  const recordingRef = useRef<Audio.Recording | null>(null);
  const scrollRef = useRef<ScrollView>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const soundRef = useRef<Audio.Sound | null>(null);
  const chunkIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttemptRef = useRef(0);
  const intentionalCloseRef = useRef(false);

  const addToHistory = useCallback(
    (type: "user" | "system" | "status", text: string) => {
      setHistory((prev) => [...prev.slice(-50), { type, text }]);
      setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 100);
    },
    [],
  );

  // ── Handle incoming WebSocket messages ──
  const handleWSMessage = useCallback(
    (event: WebSocketMessageEvent) => {
      try {
        if (typeof event.data === "string") {
          const data = JSON.parse(event.data);

          if (data.type === "tts_done") {
            console.log("[WS] TTS audio stream complete");
            return;
          }

          if (data.type === "pong") {
            console.log("[WS] Pong received (keepalive OK)");
            return;
          }

          if (data.type === "vad_reset") {
            console.log("[WS] VAD reset confirmed");
            addToHistory("status", "🔄 VAD reset");
            return;
          }

          if (data.type === "stats") {
            console.log("[WS] Server stats:", JSON.stringify(data, null, 2));
            return;
          }

          // It's a voice command result
          console.log("[WS] Voice result:", data.action, data.text || "");
          handleVoiceResult(data);
        }
        // Binary data = TTS audio chunks
        // In React Native, WebSocket binary comes as base64 or ArrayBuffer
        // We'll handle TTS via the scan/audio endpoint instead for reliability
      } catch (err) {
        console.error("[WS] Message parse error:", err);
      }
    },
    [addToHistory],
  );

  const handleVoiceResult = useCallback(
    (data: VoiceResult) => {
      if (data.text) {
        addToHistory("user", `🎤 "${data.text}"`);
      }

      switch (data.action) {
        case "scan":
          addToHistory("system", "📸 Scan triggered");
          scanEvents.emit("scan");
          break;
        case "start_continuous":
          addToHistory("system", "▶️ Continuous scanning started");
          scanEvents.emit("scan_continue");
          break;
        case "stop_continuous":
          addToHistory("system", "⏹ Scanning stopped");
          scanEvents.emit("scan_stop");
          break;
        case "switch_mode":
          addToHistory("system", `🔄 Switched to ${data.mode} mode`);
          break;
        case "follow_up":
          addToHistory(
            "system",
            data.tts_text ||
            data.response ||
            data.message ||
            "Follow-up processed",
          );
          // Play TTS for follow-up responses
          if (data.tts_text) playTTSText(data.tts_text);
          break;
        case "repeat":
          addToHistory("system", `🔁 ${data.tts_text || "Nothing to repeat"}`);
          if (data.tts_text) playTTSText(data.tts_text);
          break;
        default:
          addToHistory("system", data.message || "Command processed");
      }
    },
    [addToHistory],
  );

  // Play TTS via chat endpoint (for follow-up responses)
  const playTTSText = async (text: string) => {
    // For now, just display — TTS audio comes from scan/audio or WebSocket binary
    // Future: dedicated TTS endpoint
  };

  // ── WebSocket connection management with auto-reconnect ──
  const startPingInterval = useCallback(() => {
    if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
    pingIntervalRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action: "ping" }));
        console.log("[WS] Ping sent (keepalive)");
      }
    }, PING_INTERVAL_MS);
  }, []);

  const stopPingInterval = useCallback(() => {
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }
  }, []);

  const connectWS = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    intentionalCloseRef.current = false;

    console.log(`[WS] Connecting to ${VOICE_WS_URL}...`);
    const ws = new WebSocket(VOICE_WS_URL);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      setConnected(true);
      reconnectAttemptRef.current = 0; // Reset reconnect counter
      addToHistory("status", "🟢 Connected to voice server");
      console.log("[WS] ✅ Connected");
      startPingInterval();
    };

    ws.onmessage = handleWSMessage;

    ws.onerror = (err) => {
      console.error("[WS] ❌ Error:", err);
      setConnected(false);
    };

    ws.onclose = (event) => {
      setConnected(false);
      stopPingInterval();
      console.log(`[WS] 🔴 Closed (code=${event?.code}, intentional=${intentionalCloseRef.current})`);

      if (!intentionalCloseRef.current) {
        // Auto-reconnect with exponential backoff
        const attempt = reconnectAttemptRef.current;
        const delay = Math.min(
          RECONNECT_BASE_DELAY_MS * Math.pow(2, attempt),
          RECONNECT_MAX_DELAY_MS,
        );
        reconnectAttemptRef.current = attempt + 1;

        addToHistory("status", `🔴 Disconnected — reconnecting in ${(delay / 1000).toFixed(1)}s (attempt ${attempt + 1})`);
        console.log(`[WS] Scheduling reconnect in ${delay}ms (attempt ${attempt + 1})`);

        reconnectTimerRef.current = setTimeout(() => {
          console.log(`[WS] Reconnect attempt ${attempt + 1}...`);
          connectWS();
        }, delay);
      } else {
        addToHistory("status", "🔴 Disconnected from voice server");
      }
    };

    wsRef.current = ws;
  }, [handleWSMessage, addToHistory, startPingInterval, stopPingInterval]);

  const disconnectWS = useCallback(() => {
    intentionalCloseRef.current = true;
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    stopPingInterval();
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnected(false);
    console.log("[WS] Intentionally disconnected");
  }, [stopPingInterval]);

  // ── Recording: push-to-talk fallback (sends complete audio to /voice) ──
  const startRecording = async () => {
    if (listening) return; // Prevent double-start
    try {
      // Unload previous just in case
      if (recordingRef.current) {
        await recordingRef.current.stopAndUnloadAsync().catch(() => { });
        recordingRef.current = null;
      }

      const { granted } = await Audio.requestPermissionsAsync();
      if (!granted) {
        addToHistory("status", "⚠️ Microphone permission denied");
        return;
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY,
      );

      recordingRef.current = recording;
      setListening(true);
    } catch (err) {
      console.error("Recording start error:", err);
      addToHistory("status", "⚠️ Failed to start recording");
      setListening(false);
    }
  };

  const stopRecording = async () => {
    if (!recordingRef.current) return;

    setListening(false);
    setProcessing(true);

    try {
      // Need a tiny delay to ensure audio isn't completely empty if tapped too fast
      await new Promise((resolve) => setTimeout(resolve, 300));

      await recordingRef.current.stopAndUnloadAsync();
      const uri = recordingRef.current.getURI();
      recordingRef.current = null;

      if (uri) {
        addToHistory("user", "🎤 Voice command sent...");
        const result = await api.sendVoice(uri);
        handleVoiceResult(result);
      }
    } catch (err) {
      console.error("Recording stop error:", err);
      addToHistory("system", "⚠️ Failed to process voice command");
      recordingRef.current = null;
    } finally {
      setProcessing(false);
    }
  };

  // ── Cleanup ──
  useEffect(() => {
    return () => {
      intentionalCloseRef.current = true;
      disconnectWS();
      if (chunkIntervalRef.current) clearInterval(chunkIntervalRef.current);
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (soundRef.current) soundRef.current.unloadAsync().catch(() => { });
    };
  }, [disconnectWS]);

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Voice Control</Text>
        <Text style={styles.headerSub}>Control everything with your voice</Text>
      </View>

      {/* Connection Status */}
      <View style={styles.connectionBar}>
        <View
          style={[
            styles.connDot,
            { backgroundColor: connected ? "#00b894" : "#636e72" },
          ]}
        />
        <Text style={styles.connText}>
          {connected ? "Voice server connected" : "Not connected"}
        </Text>
        <TouchableOpacity
          style={[
            styles.connBtn,
            connected ? styles.connBtnDisconnect : styles.connBtnConnect,
          ]}
          onPress={connected ? disconnectWS : connectWS}
        >
          <Text style={styles.connBtnText}>
            {connected ? "Disconnect" : "Connect"}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Command Guide */}
      <View style={styles.guideBox}>
        <Text style={styles.guideTitle}>Voice Commands</Text>
        <View style={styles.guideGrid}>
          {[
            { cmd: '"Scan"', desc: "One-shot capture", icon: "📸" },
            { cmd: '"Scan continue"', desc: "Continuous mode", icon: "▶️" },
            { cmd: '"Scan stop"', desc: "Stop scanning", icon: "⏹" },
            { cmd: '"Medical mode"', desc: "Switch mode", icon: "💊" },
            { cmd: '"Tell me more"', desc: "Follow-up", icon: "💬" },
            { cmd: '"Repeat"', desc: "Replay last", icon: "🔁" },
          ].map((item, i) => (
            <View key={i} style={styles.guideItem}>
              <Text style={styles.guideIcon}>{item.icon}</Text>
              <View>
                <Text style={styles.guideCmd}>{item.cmd}</Text>
                <Text style={styles.guideDesc}>{item.desc}</Text>
              </View>
            </View>
          ))}
        </View>
      </View>

      {/* History */}
      <ScrollView
        ref={scrollRef}
        style={styles.history}
        contentContainerStyle={styles.historyContent}
      >
        {history.length === 0 ? (
          <Text style={styles.placeholder}>
            {connected
              ? "Listening for voice commands via WebSocket...\nSpeak naturally — VAD detects when you stop."
              : 'Tap "Connect" above, or hold the mic button below'}
          </Text>
        ) : (
          history.map((item, i) => (
            <View
              key={i}
              style={[
                styles.historyItem,
                item.type === "user"
                  ? styles.historyUser
                  : item.type === "status"
                    ? styles.historyStatus
                    : styles.historySystem,
              ]}
            >
              <Text
                style={[
                  styles.historyText,
                  item.type === "status" && styles.historyStatusText,
                ]}
              >
                {item.text}
              </Text>
            </View>
          ))
        )}
      </ScrollView>

      {/* Mic Button (push-to-talk fallback) */}
      <View style={styles.micContainer}>
        {processing && (
          <View style={styles.processingBadge}>
            <ActivityIndicator size="small" color="#6c5ce7" />
            <Text style={styles.processingText}>Processing...</Text>
          </View>
        )}
        <TouchableOpacity
          style={[styles.micBtn, listening && styles.micBtnActive]}
          onPressIn={startRecording}
          onPressOut={stopRecording}
          activeOpacity={0.8}
        >
          <Text style={styles.micIcon}>{listening ? "🔴" : "🎤"}</Text>
          <Text style={styles.micLabel}>
            {listening ? "Listening..." : "Hold to speak (fallback)"}
          </Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0a0a0f" },
  // Header
  header: { paddingHorizontal: 16, paddingTop: 16, paddingBottom: 8 },
  headerTitle: { fontSize: 24, fontWeight: "700", color: "#f0f0f5" },
  headerSub: { fontSize: 14, color: "#55556a", marginTop: 4 },
  // Connection
  connectionBar: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginHorizontal: 12,
    marginTop: 8,
    paddingHorizontal: 14,
    paddingVertical: 10,
    backgroundColor: "#12121a",
    borderRadius: 12,
  },
  connDot: { width: 8, height: 8, borderRadius: 4 },
  connText: { flex: 1, color: "#8888a0", fontSize: 13 },
  connBtn: { paddingHorizontal: 14, paddingVertical: 6, borderRadius: 8 },
  connBtnConnect: { backgroundColor: "#6c5ce7" },
  connBtnDisconnect: { backgroundColor: "#e17055" },
  connBtnText: { color: "#fff", fontSize: 12, fontWeight: "600" },
  // Guide
  guideBox: {
    marginHorizontal: 12,
    marginTop: 8,
    backgroundColor: "#12121a",
    borderRadius: 12,
    padding: 14,
  },
  guideTitle: {
    color: "#8888a0",
    fontSize: 12,
    fontWeight: "600",
    marginBottom: 10,
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  guideGrid: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
  guideItem: {
    backgroundColor: "#1a1a2e",
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 6,
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  guideIcon: { fontSize: 16 },
  guideCmd: { color: "#6c5ce7", fontSize: 13, fontWeight: "600" },
  guideDesc: { color: "#55556a", fontSize: 11, marginTop: 1 },
  // History
  history: { flex: 1, marginTop: 12 },
  historyContent: { paddingHorizontal: 12, paddingBottom: 8 },
  placeholder: {
    color: "#55556a",
    textAlign: "center",
    marginTop: 32,
    fontSize: 14,
    lineHeight: 22,
  },
  historyItem: {
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 12,
    marginBottom: 8,
    maxWidth: "85%",
  },
  historyUser: { backgroundColor: "#1a1a2e", alignSelf: "flex-end" },
  historySystem: {
    backgroundColor: "#12121a",
    borderLeftWidth: 3,
    borderLeftColor: "#6c5ce7",
    alignSelf: "flex-start",
  },
  historyStatus: {
    backgroundColor: "transparent",
    alignSelf: "center",
    paddingVertical: 4,
  },
  historyText: { color: "#f0f0f5", fontSize: 14, lineHeight: 20 },
  historyStatusText: { color: "#55556a", fontSize: 12, fontStyle: "italic" },
  // Mic
  micContainer: {
    paddingHorizontal: 12,
    paddingBottom: 16,
    paddingTop: 8,
    alignItems: "center",
  },
  processingBadge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 10,
  },
  processingText: { color: "#6c5ce7", fontSize: 13 },
  micBtn: {
    width: "100%",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    paddingVertical: 16,
    backgroundColor: "#1a1a2e",
    borderRadius: 16,
    borderWidth: 2,
    borderColor: "rgba(255,255,255,0.06)",
  },
  micBtnActive: {
    backgroundColor: "#6c5ce7",
    borderColor: "#6c5ce7",
  },
  micIcon: { fontSize: 20 },
  micLabel: { color: "#f0f0f5", fontSize: 16, fontWeight: "600" },
});
