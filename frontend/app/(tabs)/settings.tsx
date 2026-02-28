import React, { useState, useEffect } from 'react';
import {
    StyleSheet,
    View,
    Text,
    TouchableOpacity,
    ScrollView,
    SafeAreaView,
    TextInput,
    ActivityIndicator,
    Switch,
} from 'react-native';
import api, { type HealthStatus, type Mode } from '@/services/api';
import { API_BASE } from '@/services/api';

type TTSProvider = 'elevenlabs' | 'sarvam';

export default function SettingsScreen() {
    const [health, setHealth] = useState<HealthStatus | null>(null);
    const [loading, setLoading] = useState(false);
    const [mode, setMode] = useState<Mode>('medical');
    const [serverUrl, setServerUrl] = useState(API_BASE);
    const [ttsProvider, setTtsProvider] = useState<TTSProvider>('sarvam');
    const [llmProvider, setLlmProvider] = useState<'google' | 'groq'>('google');
    const [llmUpdating, setLlmUpdating] = useState(false);
    const [detectionEnabled, setDetectionEnabled] = useState(true);
    const [detectionUpdating, setDetectionUpdating] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [ttsUpdating, setTtsUpdating] = useState(false);

    useEffect(() => {
        checkHealth();
    }, []);

    const checkHealth = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await api.health();
            setHealth(data);
            setMode(data.mode as Mode);
            setTtsProvider(data.tts_provider as TTSProvider);
            setLlmProvider(data.llm_provider as 'google' | 'groq');
            setDetectionEnabled(data.detection_enabled ?? true);
        } catch (err) {
            setError('Cannot connect to backend');
            setHealth(null);
        } finally {
            setLoading(false);
        }
    };

    const handleModeSwitch = async (newMode: Mode) => {
        try {
            const result = await api.switchMode(newMode);
            if (result.status === 'ok') {
                setMode(newMode);
                await checkHealth();
            }
        } catch (err) {
            setError('Failed to switch mode');
        }
    };

    const handleServerUpdate = () => {
        api.setBaseUrl(serverUrl);
        checkHealth();
    };

    const handleTTSSwitch = async (provider: TTSProvider) => {
        setTtsUpdating(true);
        try {
            await api.updateTtsProvider(provider);
            setTtsProvider(provider);
        } catch (err) {
            setError('Failed to switch TTS provider');
        } finally {
            setTtsUpdating(false);
        }
    };

    const handleDetectionToggle = async (value: boolean) => {
        setDetectionUpdating(true);
        setDetectionEnabled(value);
        try {
            await api.toggleDetection(value);
            console.log(`[Settings] YOLO detection ${value ? 'enabled' : 'disabled'}`);
        } catch (err) {
            setDetectionEnabled(!value);
            setError('Failed to update detection setting');
        } finally {
            setDetectionUpdating(false);
        }
    };

    const handleLLMSwitch = async (provider: 'google' | 'groq') => {
        setLlmUpdating(true);
        try {
            await api.switchLlmProvider(provider);
            setLlmProvider(provider);
            await checkHealth(); // Refresh to get updated model name
        } catch (err) {
            setError('Failed to switch vision model');
        } finally {
            setLlmUpdating(false);
        }
    };

    return (
        <SafeAreaView style={styles.container}>
            <ScrollView contentContainerStyle={styles.content}>
                {/* Header */}
                <Text style={styles.title}>Settings</Text>

                {/* Connection Status */}
                <View style={styles.card}>
                    <View style={styles.cardHeader}>
                        <Text style={styles.cardTitle}>Backend Connection</Text>
                        <View style={[styles.statusDot, { backgroundColor: health ? '#00b894' : '#e17055' }]} />
                    </View>
                    {loading ? (
                        <ActivityIndicator color="#6c5ce7" style={{ marginVertical: 12 }} />
                    ) : health ? (
                        <View style={styles.statusGrid}>
                            <StatusRow label="Status" value={health.status} />
                            <StatusRow label="Mode" value={health.mode} />
                            <StatusRow label="LLM" value={`${health.llm_provider} / ${health.llm_model ?? '?'}`} />
                            <StatusRow label="TTS" value={health.tts_provider} />
                            <StatusRow label="STT" value={health.stt_provider} />
                        </View>
                    ) : (
                        <Text style={styles.errorText}>{error || 'Not connected'}</Text>
                    )}
                    <TouchableOpacity style={styles.refreshBtn} onPress={checkHealth}>
                        <Text style={styles.refreshBtnText}>🔄 Refresh</Text>
                    </TouchableOpacity>
                </View>

                {/* Server URL */}
                <View style={styles.card}>
                    <Text style={styles.cardTitle}>Server URL</Text>
                    <TextInput
                        style={styles.input}
                        value={serverUrl}
                        onChangeText={setServerUrl}
                        placeholder="https://..."
                        placeholderTextColor="#55556a"
                        autoCapitalize="none"
                        autoCorrect={false}
                    />
                    <TouchableOpacity style={styles.saveBtn} onPress={handleServerUpdate}>
                        <Text style={styles.saveBtnText}>Connect</Text>
                    </TouchableOpacity>
                </View>

                {/* Mode Switch */}
                <View style={styles.card}>
                    <Text style={styles.cardTitle}>Active Mode</Text>
                    <Text style={styles.cardSub}>Also switchable via voice: "switch to medical mode"</Text>
                    <View style={styles.modeRow}>
                        <TouchableOpacity
                            style={[styles.modeBtn, mode === 'medical' && styles.modeBtnActiveMed]}
                            onPress={() => handleModeSwitch('medical')}
                        >
                            <Text style={styles.modeBtnIcon}>💊</Text>
                            <Text style={[styles.modeBtnText, mode === 'medical' && styles.modeBtnTextActive]}>Medical</Text>
                        </TouchableOpacity>
                        <TouchableOpacity
                            style={[styles.modeBtn, mode === 'retail' && styles.modeBtnActiveRet]}
                            onPress={() => handleModeSwitch('retail')}
                        >
                            <Text style={styles.modeBtnIcon}>🛒</Text>
                            <Text style={[styles.modeBtnText, mode === 'retail' && styles.modeBtnTextActive]}>Retail</Text>
                        </TouchableOpacity>
                    </View>
                </View>

                {/* Vision Model */}
                <View style={styles.card}>
                    <View style={styles.cardHeader}>
                        <Text style={styles.cardTitle}>Vision Model</Text>
                        {llmUpdating && <ActivityIndicator color="#6c5ce7" size="small" />}
                    </View>
                    <Text style={styles.cardSub}>Choose LLM for image analysis (Tier 2)</Text>
                    <View style={styles.modeRow}>
                        <TouchableOpacity
                            style={[styles.modeBtn, llmProvider === 'google' && styles.llmBtnActiveGoogle]}
                            onPress={() => handleLLMSwitch('google')}
                            disabled={llmUpdating}
                        >
                            <Text style={styles.modeBtnIcon}>✨</Text>
                            <View>
                                <Text style={[styles.modeBtnText, llmProvider === 'google' && styles.modeBtnTextActive]}>
                                    Gemini
                                </Text>
                                <Text style={styles.ttsSubText}>2.0 Flash</Text>
                            </View>
                        </TouchableOpacity>
                        <TouchableOpacity
                            style={[styles.modeBtn, llmProvider === 'groq' && styles.llmBtnActiveGroq]}
                            onPress={() => handleLLMSwitch('groq')}
                            disabled={llmUpdating}
                        >
                            <Text style={styles.modeBtnIcon}>⚡</Text>
                            <View>
                                <Text style={[styles.modeBtnText, llmProvider === 'groq' && styles.modeBtnTextActive]}>
                                    Groq
                                </Text>
                                <Text style={styles.ttsSubText}>Llama 4 Scout</Text>
                            </View>
                        </TouchableOpacity>
                    </View>
                    {health?.llm_model && (
                        <Text style={styles.activeModelText}>Active: {health.llm_model}</Text>
                    )}
                </View>

                {/* Pipeline Config */}
                <View style={styles.card}>
                    <Text style={styles.cardTitle}>Pipeline</Text>
                    <Text style={styles.cardSub}>Configure how frames are processed</Text>

                    {/* YOLO Toggle */}
                    <View style={styles.toggleRow}>
                        <View style={styles.toggleInfo}>
                            <Text style={styles.toggleLabel}>YOLO Detection</Text>
                            <Text style={styles.toggleSub}>
                                {detectionEnabled
                                    ? '📷 → YOLO → Gemini Vision → TTS'
                                    : '📷 → Gemini Vision directly → TTS'}
                            </Text>
                        </View>
                        {detectionUpdating ? (
                            <ActivityIndicator color="#6c5ce7" size="small" />
                        ) : (
                            <Switch
                                value={detectionEnabled}
                                onValueChange={handleDetectionToggle}
                                trackColor={{ false: '#1a1a2e', true: '#6c5ce755' }}
                                thumbColor={detectionEnabled ? '#6c5ce7' : '#55556a'}
                            />
                        )}
                    </View>
                    <Text style={styles.toggleHint}>
                        {detectionEnabled
                            ? 'YOLO on: Only frames with detected objects reach Gemini. Faster, but may miss things.'
                            : 'YOLO off: Every frame goes to Gemini directly. More thorough, slightly slower.'}
                    </Text>
                </View>

                {/* TTS Provider */}
                <View style={styles.card}>
                    <View style={styles.cardHeader}>
                        <Text style={styles.cardTitle}>TTS Provider</Text>
                        {ttsUpdating && <ActivityIndicator color="#6c5ce7" size="small" />}
                    </View>
                    <Text style={styles.cardSub}>Choose text-to-speech engine for voice responses</Text>
                    <View style={styles.modeRow}>
                        <TouchableOpacity
                            style={[styles.modeBtn, ttsProvider === 'elevenlabs' && styles.ttsBtnActive]}
                            onPress={() => handleTTSSwitch('elevenlabs')}
                            disabled={ttsUpdating}
                        >
                            <Text style={styles.modeBtnIcon}>🎙️</Text>
                            <View>
                                <Text style={[styles.modeBtnText, ttsProvider === 'elevenlabs' && styles.modeBtnTextActive]}>
                                    ElevenLabs
                                </Text>
                                <Text style={styles.ttsSubText}>Natural voice</Text>
                            </View>
                        </TouchableOpacity>
                        <TouchableOpacity
                            style={[styles.modeBtn, ttsProvider === 'sarvam' && styles.ttsBtnActive]}
                            onPress={() => handleTTSSwitch('sarvam')}
                            disabled={ttsUpdating}
                        >
                            <Text style={styles.modeBtnIcon}>🇮🇳</Text>
                            <View>
                                <Text style={[styles.modeBtnText, ttsProvider === 'sarvam' && styles.modeBtnTextActive]}>
                                    Sarvam
                                </Text>
                                <Text style={styles.ttsSubText}>Indian voices</Text>
                            </View>
                        </TouchableOpacity>
                    </View>
                </View>

                {/* Info */}
                <View style={styles.card}>
                    <Text style={styles.cardTitle}>About</Text>
                    <Text style={styles.aboutText}>
                        Vision AI — Assistive system for visually impaired users.{'\n'}
                        Dual-mode: Medical (medicines, prescriptions) + Retail (products, currency).{'\n'}
                        All interaction via voice commands.{'\n\n'}
                        Voice: "scan" → one-shot | "scan continue" → continuous | "stop" → stop{'\n'}
                        "what is this" / "read this" / "check this" → all trigger a scan
                    </Text>
                </View>
            </ScrollView>
        </SafeAreaView>
    );
}

function StatusRow({ label, value }: { label: string; value: string }) {
    return (
        <View style={styles.statusRow}>
            <Text style={styles.statusLabel}>{label}</Text>
            <Text style={styles.statusValue}>{value}</Text>
        </View>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#0a0a0f' },
    content: { padding: 16, paddingBottom: 40 },
    title: { fontSize: 28, fontWeight: '700', color: '#f0f0f5', marginBottom: 20 },
    // Card
    card: { backgroundColor: '#12121a', borderRadius: 16, padding: 16, marginBottom: 16, borderWidth: 1, borderColor: 'rgba(255,255,255,0.06)' },
    cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 },
    cardTitle: { color: '#f0f0f5', fontSize: 16, fontWeight: '600' },
    cardSub: { color: '#55556a', fontSize: 12, marginTop: 4, marginBottom: 12 },
    statusDot: { width: 10, height: 10, borderRadius: 5 },
    // Status Grid
    statusGrid: { gap: 6, marginBottom: 12 },
    statusRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 4 },
    statusLabel: { color: '#8888a0', fontSize: 14 },
    statusValue: { color: '#f0f0f5', fontSize: 14, fontWeight: '500' },
    errorText: { color: '#e17055', fontSize: 14, marginVertical: 8 },
    refreshBtn: { backgroundColor: '#1a1a2e', borderRadius: 10, paddingVertical: 10, alignItems: 'center' },
    refreshBtnText: { color: '#f0f0f5', fontSize: 14, fontWeight: '500' },
    // Server URL
    input: { backgroundColor: '#1a1a2e', borderRadius: 10, paddingHorizontal: 14, paddingVertical: 12, color: '#f0f0f5', fontSize: 14, marginTop: 10, borderWidth: 1, borderColor: 'rgba(255,255,255,0.06)' },
    saveBtn: { backgroundColor: '#6c5ce7', borderRadius: 10, paddingVertical: 12, alignItems: 'center', marginTop: 10 },
    saveBtnText: { color: '#fff', fontSize: 14, fontWeight: '600' },
    // Mode
    modeRow: { flexDirection: 'row', gap: 12 },
    modeBtn: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, paddingVertical: 14, backgroundColor: '#1a1a2e', borderRadius: 12, borderWidth: 2, borderColor: 'rgba(255,255,255,0.06)' },
    modeBtnActiveMed: { borderColor: '#6c5ce7', backgroundColor: '#6c5ce711' },
    modeBtnActiveRet: { borderColor: '#00cec9', backgroundColor: '#00cec911' },
    ttsBtnActive: { borderColor: '#e17055', backgroundColor: '#e1705511' },
    modeBtnIcon: { fontSize: 20 },
    modeBtnText: { color: '#8888a0', fontSize: 15, fontWeight: '600' },
    modeBtnTextActive: { color: '#f0f0f5' },
    ttsSubText: { color: '#55556a', fontSize: 10, marginTop: 1 },
    llmBtnActiveGoogle: { borderColor: '#00b894', backgroundColor: '#00b89411' },
    llmBtnActiveGroq: { borderColor: '#fdcb6e', backgroundColor: '#fdcb6e11' },
    activeModelText: { color: '#55556a', fontSize: 11, marginTop: 8, textAlign: 'center' },
    // Toggle / Pipeline
    toggleRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginTop: 4, marginBottom: 8 },
    toggleInfo: { flex: 1, marginRight: 12 },
    toggleLabel: { color: '#f0f0f5', fontSize: 15, fontWeight: '600' },
    toggleSub: { color: '#6c5ce7', fontSize: 11, marginTop: 2, fontFamily: 'monospace' },
    toggleHint: { color: '#55556a', fontSize: 12, lineHeight: 16, marginTop: 4 },
    // About
    aboutText: { color: '#8888a0', fontSize: 13, lineHeight: 20, marginTop: 8 },
});
