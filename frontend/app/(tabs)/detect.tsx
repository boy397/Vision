import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
    StyleSheet,
    View,
    Text,
    TouchableOpacity,
    Dimensions,
    Animated,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import api, { type Detection } from '@/services/api';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

// Color palette for unique track IDs
const TRACK_COLORS = [
    '#6c5ce7', '#00cec9', '#fdcb6e', '#e17055', '#00b894',
    '#0984e3', '#e84393', '#fd79a8', '#55efc4', '#ffeaa7',
    '#74b9ff', '#a29bfe', '#fab1a0', '#81ecec', '#dfe6e9',
];

function getTrackColor(trackId: number): string {
    if (trackId < 0) return '#6c5ce7';
    return TRACK_COLORS[trackId % TRACK_COLORS.length];
}

interface TrackedDetection extends Detection {
    color: string;
}

export default function DetectScreen() {
    const [permission, requestPermission] = useCameraPermissions();
    const [tracking, setTracking] = useState(false);
    const [detections, setDetections] = useState<TrackedDetection[]>([]);
    const [fps, setFps] = useState(0);
    const [latencyMs, setLatencyMs] = useState(0);
    const [objectCount, setObjectCount] = useState(0);
    const [frameShape, setFrameShape] = useState<[number, number]>([480, 640]);
    const [totalFrames, setTotalFrames] = useState(0);

    const cameraRef = useRef<CameraView>(null);
    const trackingRef = useRef(false);
    const processingRef = useRef(false);
    const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const fpsCountRef = useRef(0);
    const fpsIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const pulseAnim = useRef(new Animated.Value(1)).current;

    // Pulse animation for tracking indicator
    useEffect(() => {
        if (tracking) {
            const pulse = Animated.loop(
                Animated.sequence([
                    Animated.timing(pulseAnim, { toValue: 0.3, duration: 500, useNativeDriver: true }),
                    Animated.timing(pulseAnim, { toValue: 1, duration: 500, useNativeDriver: true }),
                ])
            );
            pulse.start();
            return () => pulse.stop();
        }
        pulseAnim.setValue(1);
    }, [tracking]);

    const captureAndDetect = useCallback(async () => {
        if (!cameraRef.current || processingRef.current || !trackingRef.current) return;
        processingRef.current = true;

        try {
            const photo = await cameraRef.current.takePictureAsync({
                quality: 0.5,
                base64: false,
                skipProcessing: true,
            });
            if (!photo?.uri) return;

            const start = Date.now();
            const result = await api.detectObjects(photo.uri);
            const elapsed = Date.now() - start;

            if (!trackingRef.current) return;

            const tracked: TrackedDetection[] = result.detections.map(d => ({
                ...d,
                color: getTrackColor(d.track_id ?? -1),
            }));

            setDetections(tracked);
            setLatencyMs(elapsed);
            setObjectCount(tracked.length);
            setFrameShape(result.frame_shape);
            setTotalFrames(p => p + 1);
            fpsCountRef.current += 1;
        } catch (err) {
            // Ignore errors silently for demo
        } finally {
            processingRef.current = false;
        }
    }, []);

    const startTracking = useCallback(() => {
        trackingRef.current = true;
        setTracking(true);
        setTotalFrames(0);
        setDetections([]);
        fpsCountRef.current = 0;

        // Capture at ~3 FPS (every 333ms)
        intervalRef.current = setInterval(() => {
            captureAndDetect();
        }, 333);

        // FPS counter
        fpsIntervalRef.current = setInterval(() => {
            setFps(fpsCountRef.current);
            fpsCountRef.current = 0;
        }, 1000);
    }, [captureAndDetect]);

    const stopTracking = useCallback(() => {
        trackingRef.current = false;
        setTracking(false);
        if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
        if (fpsIntervalRef.current) { clearInterval(fpsIntervalRef.current); fpsIntervalRef.current = null; }
        setFps(0);
        setDetections([]);
    }, []);

    // Cleanup
    useEffect(() => {
        return () => {
            trackingRef.current = false;
            if (intervalRef.current) clearInterval(intervalRef.current);
            if (fpsIntervalRef.current) clearInterval(fpsIntervalRef.current);
        };
    }, []);

    if (!permission) return <View style={styles.container} />;

    if (!permission.granted) {
        return (
            <View style={styles.container}>
                <View style={styles.permissionBox}>
                    <Text style={styles.permTitle}>📷 Camera Access</Text>
                    <Text style={styles.permText}>Camera is needed for real-time object tracking.</Text>
                    <TouchableOpacity style={styles.permBtn} onPress={requestPermission}>
                        <Text style={styles.permBtnText}>Grant Access</Text>
                    </TouchableOpacity>
                </View>
            </View>
        );
    }

    const scaleX = SCREEN_WIDTH / frameShape[1];
    const scaleY = SCREEN_HEIGHT / frameShape[0];

    return (
        <View style={styles.container}>
            {/* Full-screen camera */}
            <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back" />

            {/* Bounding boxes */}
            {detections.map((det, i) => {
                const [x1, y1, x2, y2] = det.bbox;
                const left = x1 * scaleX;
                const top = y1 * scaleY;
                const w = Math.max((x2 - x1) * scaleX, 20);
                const h = Math.max((y2 - y1) * scaleY, 20);
                const color = det.color;

                return (
                    <React.Fragment key={det.track_id != null && det.track_id >= 0 ? `t${det.track_id}` : `d${i}`}>
                        {/* Box */}
                        <View
                            style={{
                                position: 'absolute', left, top, width: w, height: h,
                                borderWidth: 2, borderColor: color, borderRadius: 4,
                            }}
                        >
                            {/* Corner accents */}
                            <View style={{ position: 'absolute', top: -1, left: -1, width: 14, height: 14, borderTopWidth: 3, borderLeftWidth: 3, borderColor: color, borderTopLeftRadius: 3 }} />
                            <View style={{ position: 'absolute', top: -1, right: -1, width: 14, height: 14, borderTopWidth: 3, borderRightWidth: 3, borderColor: color, borderTopRightRadius: 3 }} />
                            <View style={{ position: 'absolute', bottom: -1, left: -1, width: 14, height: 14, borderBottomWidth: 3, borderLeftWidth: 3, borderColor: color, borderBottomLeftRadius: 3 }} />
                            <View style={{ position: 'absolute', bottom: -1, right: -1, width: 14, height: 14, borderBottomWidth: 3, borderRightWidth: 3, borderColor: color, borderBottomRightRadius: 3 }} />
                        </View>
                        {/* Label */}
                        <View style={{
                            position: 'absolute', left, top: Math.max(top - 26, 0),
                            backgroundColor: color, paddingHorizontal: 7, paddingVertical: 3, borderRadius: 4,
                        }}>
                            <Text style={{ color: '#fff', fontSize: 12, fontWeight: '800' }}>
                                {det.class_name} {(det.confidence * 100).toFixed(0)}%
                                {det.track_id != null && det.track_id >= 0 ? ` #${det.track_id}` : ''}
                            </Text>
                        </View>
                    </React.Fragment>
                );
            })}

            {/* Top HUD — glass overlay */}
            <View style={styles.topHud}>
                <View style={styles.hudRow}>
                    <View style={[styles.badge, { backgroundColor: tracking ? '#00b89440' : '#636e7240' }]}>
                        <Animated.View style={[styles.dot, { backgroundColor: tracking ? '#00b894' : '#636e72', opacity: tracking ? pulseAnim : 1 }]} />
                        <Text style={[styles.badgeText, { color: tracking ? '#00b894' : '#aaa' }]}>
                            {tracking ? 'LIVE' : 'IDLE'}
                        </Text>
                    </View>
                    {tracking && (
                        <>
                            <View style={[styles.badge, { backgroundColor: '#0984e340' }]}>
                                <Text style={[styles.badgeText, { color: '#74b9ff' }]}>{fps} FPS</Text>
                            </View>
                            <View style={[styles.badge, {
                                backgroundColor: latencyMs < 150 ? '#00b89440' : latencyMs < 400 ? '#fdcb6e40' : '#e1705540',
                            }]}>
                                <Text style={[styles.badgeText, {
                                    color: latencyMs < 150 ? '#55efc4' : latencyMs < 400 ? '#ffeaa7' : '#fab1a0',
                                }]}>{latencyMs}ms</Text>
                            </View>
                            <View style={[styles.badge, { backgroundColor: '#a29bfe40' }]}>
                                <Text style={[styles.badgeText, { color: '#a29bfe' }]}>{objectCount} obj</Text>
                            </View>
                        </>
                    )}
                </View>
            </View>

            {/* Bottom HUD */}
            <View style={styles.bottomHud}>
                {/* Detection legend */}
                {detections.length > 0 && (
                    <View style={styles.legendRow}>
                        {detections.map((det, i) => (
                            <View key={i} style={[styles.legendChip, { borderColor: det.color }]}>
                                <View style={[styles.legendDot, { backgroundColor: det.color }]} />
                                <Text style={styles.legendText}>
                                    {det.class_name}{det.track_id != null && det.track_id >= 0 ? ` #${det.track_id}` : ''}
                                </Text>
                            </View>
                        ))}
                    </View>
                )}

                {/* Big start/stop button */}
                <TouchableOpacity
                    style={[styles.bigButton, { backgroundColor: tracking ? '#e17055' : '#00b894' }]}
                    onPress={tracking ? stopTracking : startTracking}
                    activeOpacity={0.8}
                >
                    <Text style={styles.bigButtonIcon}>{tracking ? '⏹' : '▶'}</Text>
                    <Text style={styles.bigButtonText}>
                        {tracking ? 'Stop Tracking' : 'Start Tracking'}
                    </Text>
                    {totalFrames > 0 && (
                        <Text style={styles.frameCounter}>{totalFrames} frames</Text>
                    )}
                </TouchableOpacity>
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#000' },
    // Permission
    permissionBox: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 32 },
    permTitle: { fontSize: 24, fontWeight: '700', color: '#f0f0f5', marginBottom: 12 },
    permText: { fontSize: 16, color: '#8888a0', textAlign: 'center', marginBottom: 24 },
    permBtn: { backgroundColor: '#00b894', paddingHorizontal: 32, paddingVertical: 14, borderRadius: 12 },
    permBtnText: { color: '#fff', fontSize: 16, fontWeight: '600' },
    // Top HUD
    topHud: {
        position: 'absolute', top: 0, left: 0, right: 0,
        paddingTop: 54, paddingHorizontal: 14, paddingBottom: 10,
        backgroundColor: 'rgba(0,0,0,0.5)',
    },
    hudRow: { flexDirection: 'row', alignItems: 'center', gap: 8, flexWrap: 'wrap' },
    badge: {
        flexDirection: 'row', alignItems: 'center', gap: 5,
        paddingHorizontal: 10, paddingVertical: 5, borderRadius: 20,
    },
    dot: { width: 8, height: 8, borderRadius: 4 },
    badgeText: { fontSize: 13, fontWeight: '800', letterSpacing: 0.8, fontVariant: ['tabular-nums'] },
    // Bottom HUD
    bottomHud: {
        position: 'absolute', bottom: 0, left: 0, right: 0,
        paddingBottom: 40, paddingHorizontal: 16, paddingTop: 14,
        backgroundColor: 'rgba(0,0,0,0.55)',
    },
    legendRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginBottom: 12 },
    legendChip: {
        flexDirection: 'row', alignItems: 'center', gap: 5,
        paddingHorizontal: 8, paddingVertical: 4, borderRadius: 8,
        borderWidth: 1, backgroundColor: 'rgba(0,0,0,0.4)',
    },
    legendDot: { width: 8, height: 8, borderRadius: 4 },
    legendText: { color: '#f0f0f5', fontSize: 11, fontWeight: '600' },
    bigButton: {
        flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
        gap: 10, paddingVertical: 16, borderRadius: 16,
    },
    bigButtonIcon: { fontSize: 18 },
    bigButtonText: { color: '#fff', fontSize: 17, fontWeight: '700', letterSpacing: 0.5 },
    frameCounter: { color: 'rgba(255,255,255,0.5)', fontSize: 12, marginLeft: 4, fontVariant: ['tabular-nums'] },
});
