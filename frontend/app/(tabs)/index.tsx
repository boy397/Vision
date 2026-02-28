import React, { useState, useRef, useCallback } from 'react';
import {
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  SafeAreaView,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import api, { type ScanResult, type Detection } from '@/services/api';

export default function ScannerScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [mode, setMode] = useState<'medical' | 'retail'>('medical');
  const cameraRef = useRef<CameraView>(null);

  const handleScan = useCallback(async () => {
    if (!cameraRef.current || scanning) return;

    setScanning(true);
    const start = Date.now();

    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
        base64: false,
      });

      if (photo?.uri) {
        const data = await api.scanFromUri(photo.uri);
        setResult(data);
        setLatency(Date.now() - start);
        if (data.mode) setMode(data.mode as 'medical' | 'retail');
      }
    } catch (err) {
      console.error('Scan error:', err);
      setResult({
        detections: [],
        analysis: {},
        tts_text: 'Connection error — is the backend running?',
        mode,
        state_changed: false,
      });
    } finally {
      setScanning(false);
    }
  }, [scanning, mode]);

  if (!permission) return <View style={styles.container} />;

  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.permissionBox}>
          <Text style={styles.permissionTitle}>📷 Camera Access</Text>
          <Text style={styles.permissionText}>
            The camera is needed to scan medicines and products.
          </Text>
          <TouchableOpacity style={styles.permissionBtn} onPress={requestPermission}>
            <Text style={styles.permissionBtnText}>Grant Access</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      {/* Camera */}
      <View style={styles.cameraWrap}>
        <CameraView
          ref={cameraRef}
          style={styles.camera}
          facing="back"
        />
        {scanning && (
          <View style={styles.scanOverlay}>
            <View style={[styles.scanLine, { backgroundColor: mode === 'medical' ? '#6c5ce7' : '#00cec9' }]} />
          </View>
        )}
        {/* Mode Badge */}
        <View style={[styles.modeBadge, { backgroundColor: mode === 'medical' ? '#6c5ce722' : '#00cec922' }]}>
          <Text style={styles.modeBadgeText}>
            {mode === 'medical' ? '💊 Medical' : '🛒 Retail'}
          </Text>
        </View>
      </View>

      {/* Scan Button */}
      <TouchableOpacity
        style={[styles.scanBtn, { borderColor: mode === 'medical' ? '#6c5ce7' : '#00cec9' }]}
        onPress={handleScan}
        disabled={scanning}
        activeOpacity={0.7}
      >
        {scanning ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.scanBtnText}>📸 Scan</Text>
        )}
      </TouchableOpacity>

      {/* Results */}
      <ScrollView style={styles.results} contentContainerStyle={styles.resultsContent}>
        {latency !== null && (
          <View style={styles.latencyBadge}>
            <Text style={styles.latencyText}>{latency}ms</Text>
          </View>
        )}

        {result ? (
          <>
            {/* TTS Text */}
            {result.tts_text ? (
              <View style={[styles.ttsBox, { borderColor: mode === 'medical' ? '#6c5ce7' : '#00cec9' }]}>
                <Text style={styles.ttsLabel}>🔊 Response</Text>
                <Text style={styles.ttsText}>{result.tts_text}</Text>
              </View>
            ) : null}

            {/* Detections */}
            {result.detections.map((det: Detection, i: number) => (
              <View key={i} style={[styles.detectionItem, { borderLeftColor: mode === 'medical' ? '#6c5ce7' : '#00cec9' }]}>
                <Text style={styles.detLabel}>{det.class_name}</Text>
                <Text style={styles.detConf}>{(det.confidence * 100).toFixed(1)}%</Text>
              </View>
            ))}

            {/* Analysis */}
            {result.analysis && !result.analysis.error && Object.keys(result.analysis).length > 0 && (
              <View style={styles.analysisBox}>
                <Text style={styles.analysisTitle}>📋 Analysis</Text>
                {Object.entries(result.analysis).map(([key, val]) => {
                  if (key === 'confidence' || key === 'raw_text_detected') return null;
                  if (val && typeof val === 'object') {
                    return Object.entries(val as Record<string, any>).map(([k, v]) =>
                      v ? (
                        <Text key={`${key}.${k}`} style={styles.analysisLine}>
                          <Text style={styles.analysisKey}>{k}: </Text>
                          {String(v)}
                        </Text>
                      ) : null
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
          <Text style={styles.placeholder}>Point camera at a medicine or product and tap Scan</Text>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0a0a0f' },
  // Permission
  permissionBox: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 32 },
  permissionTitle: { fontSize: 24, fontWeight: '700', color: '#f0f0f5', marginBottom: 12 },
  permissionText: { fontSize: 16, color: '#8888a0', textAlign: 'center', marginBottom: 24 },
  permissionBtn: { backgroundColor: '#6c5ce7', paddingHorizontal: 32, paddingVertical: 14, borderRadius: 12 },
  permissionBtnText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  // Camera
  cameraWrap: { height: 280, marginHorizontal: 12, marginTop: 8, borderRadius: 16, overflow: 'hidden', position: 'relative' },
  camera: { flex: 1 },
  scanOverlay: { ...StyleSheet.absoluteFillObject, justifyContent: 'center' },
  scanLine: { height: 2, opacity: 0.8 },
  modeBadge: { position: 'absolute', top: 12, right: 12, paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20 },
  modeBadgeText: { color: '#f0f0f5', fontSize: 13, fontWeight: '600' },
  // Scan Button
  scanBtn: {
    marginHorizontal: 12, marginTop: 12, paddingVertical: 14,
    backgroundColor: '#1a1a2e', borderRadius: 12, borderWidth: 1.5,
    alignItems: 'center',
  },
  scanBtnText: { color: '#f0f0f5', fontSize: 16, fontWeight: '600' },
  // Results
  results: { flex: 1, marginTop: 12 },
  resultsContent: { paddingHorizontal: 12, paddingBottom: 20 },
  latencyBadge: { alignSelf: 'flex-end', backgroundColor: '#12121a', paddingHorizontal: 10, paddingVertical: 3, borderRadius: 8, marginBottom: 8 },
  latencyText: { color: '#55556a', fontSize: 12, fontVariant: ['tabular-nums'] },
  placeholder: { color: '#55556a', textAlign: 'center', marginTop: 32, fontSize: 14 },
  // TTS Box
  ttsBox: { backgroundColor: '#12121a', borderRadius: 12, padding: 14, borderLeftWidth: 3, marginBottom: 10 },
  ttsLabel: { color: '#8888a0', fontSize: 12, fontWeight: '600', marginBottom: 6 },
  ttsText: { color: '#f0f0f5', fontSize: 15, lineHeight: 22 },
  // Detection Item
  detectionItem: { backgroundColor: '#12121a', borderRadius: 8, padding: 12, marginBottom: 8, borderLeftWidth: 3, flexDirection: 'row', justifyContent: 'space-between' },
  detLabel: { color: '#f0f0f5', fontWeight: '600', fontSize: 14 },
  detConf: { color: '#00b894', fontSize: 13, fontVariant: ['tabular-nums'] },
  // Analysis
  analysisBox: { backgroundColor: '#12121a', borderRadius: 12, padding: 14, marginTop: 4 },
  analysisTitle: { color: '#00b894', fontWeight: '600', fontSize: 13, marginBottom: 8 },
  analysisLine: { color: '#8888a0', fontSize: 13, lineHeight: 20, marginBottom: 2 },
  analysisKey: { color: '#f0f0f5', fontWeight: '600' },
});
