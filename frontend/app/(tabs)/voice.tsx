import React, { useState, useRef } from 'react';
import {
    StyleSheet,
    View,
    Text,
    TouchableOpacity,
    ScrollView,
    SafeAreaView,
    ActivityIndicator,
} from 'react-native';
import { Audio } from 'expo-av';
import api, { type VoiceResult } from '@/services/api';

export default function VoiceScreen() {
    const [recording, setRecording] = useState(false);
    const [processing, setProcessing] = useState(false);
    const [history, setHistory] = useState<Array<{ type: 'user' | 'system'; text: string }>>([]);
    const recordingRef = useRef<Audio.Recording | null>(null);
    const scrollRef = useRef<ScrollView>(null);

    const startRecording = async () => {
        try {
            const { granted } = await Audio.requestPermissionsAsync();
            if (!granted) return;

            await Audio.setAudioModeAsync({
                allowsRecordingIOS: true,
                playsInSilentModeIOS: true,
            });

            const { recording } = await Audio.Recording.createAsync(
                Audio.RecordingOptionsPresets.HIGH_QUALITY
            );

            recordingRef.current = recording;
            setRecording(true);
        } catch (err) {
            console.error('Recording start error:', err);
        }
    };

    const stopRecording = async () => {
        if (!recordingRef.current) return;

        setRecording(false);
        setProcessing(true);

        try {
            await recordingRef.current.stopAndUnloadAsync();
            const uri = recordingRef.current.getURI();
            recordingRef.current = null;

            if (uri) {
                addToHistory('user', '🎤 Voice command sent...');
                const result = await api.sendVoice(uri);
                handleVoiceResult(result);
            }
        } catch (err) {
            console.error('Recording stop error:', err);
            addToHistory('system', '⚠️ Failed to process voice command');
        } finally {
            setProcessing(false);
        }
    };

    const handleVoiceResult = (data: VoiceResult) => {
        switch (data.action) {
            case 'scan':
                addToHistory('system', '📸 Scan triggered — switch to Scanner tab');
                break;
            case 'switch_mode':
                addToHistory('system', `🔄 Switched to ${data.mode} mode`);
                break;
            case 'start_continuous':
                addToHistory('system', '▶️ Continuous mode started');
                break;
            case 'stop_continuous':
                addToHistory('system', '⏹ Continuous mode stopped');
                break;
            case 'follow_up':
                addToHistory('system', data.tts_text || data.response || data.message || 'Follow-up processed');
                break;
            case 'repeat':
                addToHistory('system', `🔁 ${data.tts_text || 'Nothing to repeat'}`);
                break;
            default:
                addToHistory('system', data.message || 'Command processed');
        }
    };

    const addToHistory = (type: 'user' | 'system', text: string) => {
        setHistory(prev => [...prev, { type, text }]);
        setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 100);
    };

    return (
        <SafeAreaView style={styles.container}>
            {/* Header */}
            <View style={styles.header}>
                <Text style={styles.headerTitle}>Voice Control</Text>
                <Text style={styles.headerSub}>All commands are voice-driven</Text>
            </View>

            {/* Command Guide */}
            <View style={styles.guideBox}>
                <Text style={styles.guideTitle}>Commands</Text>
                <View style={styles.guideGrid}>
                    {[
                        { cmd: '"Scan this"', desc: 'One-shot scan' },
                        { cmd: '"Start watching"', desc: 'Continuous mode' },
                        { cmd: '"Stop"', desc: 'Stop scanning' },
                        { cmd: '"Medicine mode"', desc: 'Switch to medical' },
                        { cmd: '"Retail mode"', desc: 'Switch to retail' },
                        { cmd: '"Tell me more"', desc: 'Follow-up chat' },
                        { cmd: '"Repeat"', desc: 'Replay last' },
                    ].map((item, i) => (
                        <View key={i} style={styles.guideItem}>
                            <Text style={styles.guideCmd}>{item.cmd}</Text>
                            <Text style={styles.guideDesc}>{item.desc}</Text>
                        </View>
                    ))}
                </View>
            </View>

            {/* History */}
            <ScrollView ref={scrollRef} style={styles.history} contentContainerStyle={styles.historyContent}>
                {history.length === 0 ? (
                    <Text style={styles.placeholder}>Hold the mic button and speak a command</Text>
                ) : (
                    history.map((item, i) => (
                        <View
                            key={i}
                            style={[
                                styles.historyItem,
                                item.type === 'user' ? styles.historyUser : styles.historySystem,
                            ]}
                        >
                            <Text style={styles.historyText}>{item.text}</Text>
                        </View>
                    ))
                )}
            </ScrollView>

            {/* Mic Button */}
            <View style={styles.micContainer}>
                {processing && (
                    <View style={styles.processingBadge}>
                        <ActivityIndicator size="small" color="#6c5ce7" />
                        <Text style={styles.processingText}>Processing...</Text>
                    </View>
                )}
                <TouchableOpacity
                    style={[styles.micBtn, recording && styles.micBtnActive]}
                    onPressIn={startRecording}
                    onPressOut={stopRecording}
                    activeOpacity={0.8}
                >
                    <Text style={styles.micIcon}>{recording ? '🔴' : '🎤'}</Text>
                    <Text style={styles.micLabel}>
                        {recording ? 'Listening...' : 'Hold to speak'}
                    </Text>
                </TouchableOpacity>
            </View>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#0a0a0f' },
    // Header
    header: { paddingHorizontal: 16, paddingTop: 16, paddingBottom: 8 },
    headerTitle: { fontSize: 24, fontWeight: '700', color: '#f0f0f5' },
    headerSub: { fontSize: 14, color: '#55556a', marginTop: 4 },
    // Guide
    guideBox: { marginHorizontal: 12, marginTop: 8, backgroundColor: '#12121a', borderRadius: 12, padding: 14 },
    guideTitle: { color: '#8888a0', fontSize: 12, fontWeight: '600', marginBottom: 10, textTransform: 'uppercase', letterSpacing: 0.5 },
    guideGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
    guideItem: { backgroundColor: '#1a1a2e', borderRadius: 8, paddingHorizontal: 10, paddingVertical: 6 },
    guideCmd: { color: '#6c5ce7', fontSize: 13, fontWeight: '600' },
    guideDesc: { color: '#55556a', fontSize: 11, marginTop: 2 },
    // History
    history: { flex: 1, marginTop: 12 },
    historyContent: { paddingHorizontal: 12, paddingBottom: 8 },
    placeholder: { color: '#55556a', textAlign: 'center', marginTop: 32, fontSize: 14 },
    historyItem: { paddingHorizontal: 14, paddingVertical: 10, borderRadius: 12, marginBottom: 8, maxWidth: '85%' },
    historyUser: { backgroundColor: '#1a1a2e', alignSelf: 'flex-end' },
    historySystem: { backgroundColor: '#12121a', borderLeftWidth: 3, borderLeftColor: '#6c5ce7', alignSelf: 'flex-start' },
    historyText: { color: '#f0f0f5', fontSize: 14, lineHeight: 20 },
    // Mic
    micContainer: { paddingHorizontal: 12, paddingBottom: 16, paddingTop: 8, alignItems: 'center' },
    processingBadge: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 10 },
    processingText: { color: '#6c5ce7', fontSize: 13 },
    micBtn: {
        width: '100%', flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
        gap: 10, paddingVertical: 16, backgroundColor: '#1a1a2e',
        borderRadius: 16, borderWidth: 2, borderColor: 'rgba(255,255,255,0.06)',
    },
    micBtnActive: {
        backgroundColor: '#6c5ce7', borderColor: '#6c5ce7',
    },
    micIcon: { fontSize: 20 },
    micLabel: { color: '#f0f0f5', fontSize: 16, fontWeight: '600' },
});
