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
import api, { type HealthStatus, type Mode, type ModelOption } from '@/services/api';
import { API_BASE } from '@/services/api';

type TTSProvider = 'elevenlabs' | 'sarvam';
type LLMProvider = 'google' | 'groq' | 'vllm';

export default function SettingsScreen() {
    const [health, setHealth] = useState<HealthStatus | null>(null);
    const [loading, setLoading] = useState(false);
    const [mode, setMode] = useState<Mode>('medical');
    const [serverUrl, setServerUrl] = useState(API_BASE);
    const [ttsProvider, setTtsProvider] = useState<TTSProvider>('sarvam');
    const [llmProvider, setLlmProvider] = useState<LLMProvider>('groq');
    const [llmUpdating, setLlmUpdating] = useState(false);
    const [detectionEnabled, setDetectionEnabled] = useState(true);
    const [detectionUpdating, setDetectionUpdating] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [ttsUpdating, setTtsUpdating] = useState(false);
    const [availableModels, setAvailableModels] = useState<ModelOption[]>([]);
    const [activeModel, setActiveModel] = useState<string>('');
    const [modelSwitching, setModelSwitching] = useState<string | null>(null); // model id being switched to

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
            setLlmProvider(data.llm_provider as LLMProvider);
            setDetectionEnabled(data.detection_enabled ?? true);
            setActiveModel(data.llm_model ?? '');
            if (data.available_models) {
                setAvailableModels(data.available_models);
            }
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

    const handleLLMSwitch = async (provider: LLMProvider) => {
        setLlmUpdating(true);
        try {
            const result = await api.switchLlmProvider(provider);
            setLlmProvider(provider);
            if (result.available_models) {
                setAvailableModels(result.available_models);
            }
            setActiveModel(result.llm_model ?? '');
            await checkHealth();
        } catch (err) {
            setError('Failed to switch vision model');
        } finally {
            setLlmUpdating(false);
        }
    };

    const handleModelSwitch = async (modelId: string) => {
        if (modelId === activeModel) return;
        setModelSwitching(modelId);
        try {
            const result = await api.switchLlmModel(llmProvider, modelId);
            if (result.status === 'ok') {
                setActiveModel(modelId);
            }
        } catch (err) {
            setError('Failed to switch model');
        } finally {
            setModelSwitching(null);
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

                {/* Vision Model Provider */}
                <View style={styles.card}>
                    <View style={styles.cardHeader}>
                        <Text style={styles.cardTitle}>Vision Model</Text>
                        {llmUpdating && <ActivityIndicator color="#6c5ce7" size="small" />}
                    </View>
                    <Text style={styles.cardSub}>Choose LLM provider for image analysis</Text>
                    <View style={styles.modeRow}>
                        <TouchableOpacity
                            style={[styles.modeBtn, llmProvider === 'google' && styles.llmBtnActiveGoogle]}
                            onPress={() => handleLLMSwitch('google')}
                            disabled={llmUpdating}
                        >
                            <Text style={styles.modeBtnIcon}>✨</Text>
                            <Text style={[styles.modeBtnText, llmProvider === 'google' && styles.modeBtnTextActive]}>Gemini</Text>
                        </TouchableOpacity>
                        <TouchableOpacity
                            style={[styles.modeBtn, llmProvider === 'groq' && styles.llmBtnActiveGroq]}
                            onPress={() => handleLLMSwitch('groq')}
                            disabled={llmUpdating}
                        >
                            <Text style={styles.modeBtnIcon}>⚡</Text>
                            <Text style={[styles.modeBtnText, llmProvider === 'groq' && styles.modeBtnTextActive]}>Groq</Text>
                        </TouchableOpacity>
                        <TouchableOpacity
                            style={[styles.modeBtn, llmProvider === 'vllm' && styles.llmBtnActiveVllm]}
                            onPress={() => handleLLMSwitch('vllm')}
                            disabled={llmUpdating}
                        >
                            <Text style={styles.modeBtnIcon}>🧠</Text>
                            <Text style={[styles.modeBtnText, llmProvider === 'vllm' && styles.modeBtnTextActive]}>vLLM</Text>
                        </TouchableOpacity>
                    </View>
                </View>

                {/* Model Picker — driven by config.yml available_models */}
                {availableModels.length > 0 && (
                    <View style={styles.card}>
                        <Text style={styles.cardTitle}>
                            {llmProvider === 'groq' ? '⚡' : llmProvider === 'google' ? '✨' : '🧠'}{' '}
                            {llmProvider.charAt(0).toUpperCase() + llmProvider.slice(1)} Models
                        </Text>
                        <Text style={styles.cardSub}>
                            Tap to switch • defined in config.yml
                        </Text>
                        {availableModels.map((m) => {
                            const isActive = m.id === activeModel;
                            const isSwitching = modelSwitching === m.id;
                            return (
                                <TouchableOpacity
                                    key={m.id}
                                    style={[
                                        styles.modelItem,
                                        isActive && styles.modelItemActive,
                                    ]}
                                    onPress={() => handleModelSwitch(m.id)}
                                    disabled={isSwitching || isActive}
                                >
                                    <View style={styles.modelInfo}>
                                        <View style={styles.modelNameRow}>
                                            <Text style={[styles.modelName, isActive && styles.modelNameActive]}>
                                                {m.name}
                                            </Text>
                                            {m.vision && (
                                                <View style={styles.visionBadge}>
                                                    <Text style={styles.visionBadgeText}>👁 vision</Text>
                                                </View>
                                            )}
                                        </View>
                                        <Text style={styles.modelDesc}>{m.description}</Text>
                                        <Text style={styles.modelId}>{m.id}</Text>
                                    </View>
                                    {isSwitching ? (
                                        <ActivityIndicator color="#6c5ce7" size="small" />
                                    ) : isActive ? (
                                        <View style={styles.activeDot} />
                                    ) : (
                                        <View style={styles.inactiveDot} />
                                    )}
                                </TouchableOpacity>
                            );
                        })}
                    </View>
                )}

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
    llmBtnActiveVllm: { borderColor: '#e84393', backgroundColor: '#e8439311' },
    // Model picker
    modelItem: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingVertical: 12,
        paddingHorizontal: 14,
        backgroundColor: '#1a1a2e',
        borderRadius: 12,
        marginBottom: 8,
        borderWidth: 1.5,
        borderColor: 'rgba(255,255,255,0.04)',
    },
    modelItemActive: {
        borderColor: '#fdcb6e',
        backgroundColor: '#fdcb6e08',
    },
    modelInfo: { flex: 1, marginRight: 12 },
    modelNameRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
    modelName: { color: '#8888a0', fontSize: 14, fontWeight: '600' },
    modelNameActive: { color: '#f0f0f5' },
    modelDesc: { color: '#55556a', fontSize: 11, marginTop: 2 },
    modelId: { color: '#33334a', fontSize: 10, marginTop: 2, fontFamily: 'monospace' },
    visionBadge: {
        backgroundColor: '#6c5ce722',
        paddingHorizontal: 6,
        paddingVertical: 2,
        borderRadius: 6,
    },
    visionBadgeText: { color: '#6c5ce7', fontSize: 9, fontWeight: '600' },
    activeDot: { width: 12, height: 12, borderRadius: 6, backgroundColor: '#fdcb6e' },
    inactiveDot: { width: 12, height: 12, borderRadius: 6, borderWidth: 1.5, borderColor: '#33334a' },
    // Toggle / Pipeline
    toggleRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginTop: 4, marginBottom: 8 },
    toggleInfo: { flex: 1, marginRight: 12 },
    toggleLabel: { color: '#f0f0f5', fontSize: 15, fontWeight: '600' },
    toggleSub: { color: '#6c5ce7', fontSize: 11, marginTop: 2, fontFamily: 'monospace' },
    toggleHint: { color: '#55556a', fontSize: 12, lineHeight: 16, marginTop: 4 },
    // About
    aboutText: { color: '#8888a0', fontSize: 13, lineHeight: 20, marginTop: 8 },
});
