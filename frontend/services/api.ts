/**
 * API configuration for connecting to the Vision backend.
 * Change API_BASE to your backend server's address.
 */

// For Android emulator use 10.0.2.2, for physical device use your machine's LAN IP
export const API_BASE = 'http://10.0.2.2:8000';
export const WS_URL = 'ws://10.0.2.2:8000/stream';

export type Mode = 'medical' | 'retail';

export interface Detection {
    class_name: string;
    confidence: number;
    bbox: number[];
}

export interface ScanResult {
    detections: Detection[];
    analysis: Record<string, any>;
    tts_text: string;
    mode: string;
    state_changed: boolean;
}

export interface VoiceResult {
    action: string;
    text?: string;
    message?: string;
    mode?: string;
    tts_text?: string;
    response?: string;
}

export interface HealthStatus {
    status: string;
    mode: string;
    llm_provider: string;
    tts_provider: string;
    stt_provider: string;
}

// ── API Client ──

class VisionAPI {
    private base: string;

    constructor(baseUrl: string = API_BASE) {
        this.base = baseUrl;
    }

    setBaseUrl(url: string) {
        this.base = url;
    }

    async health(): Promise<HealthStatus> {
        const res = await fetch(`${this.base}/health`);
        return res.json();
    }

    async getConfig(): Promise<Record<string, any>> {
        const res = await fetch(`${this.base}/config`);
        return res.json();
    }

    async switchMode(mode: Mode): Promise<{ status: string; mode: string; message: string }> {
        const res = await fetch(`${this.base}/config/mode`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode }),
        });
        return res.json();
    }

    async scan(imageBase64: string): Promise<ScanResult> {
        // Convert base64 to blob-like FormData
        const formData = new FormData();
        formData.append('image', {
            uri: imageBase64,
            type: 'image/jpeg',
            name: 'capture.jpg',
        } as any);

        const res = await fetch(`${this.base}/scan`, {
            method: 'POST',
            body: formData,
        });
        return res.json();
    }

    async scanFromUri(imageUri: string): Promise<ScanResult> {
        const formData = new FormData();
        formData.append('image', {
            uri: imageUri,
            type: 'image/jpeg',
            name: 'capture.jpg',
        } as any);

        const res = await fetch(`${this.base}/scan`, {
            method: 'POST',
            body: formData,
        });
        return res.json();
    }

    async sendVoice(audioUri: string): Promise<VoiceResult> {
        const formData = new FormData();
        formData.append('audio', {
            uri: audioUri,
            type: 'audio/wav',
            name: 'voice.wav',
        } as any);

        const res = await fetch(`${this.base}/voice`, {
            method: 'POST',
            body: formData,
        });
        return res.json();
    }

    async chat(question: string): Promise<VoiceResult> {
        const res = await fetch(`${this.base}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });
        return res.json();
    }
}

export const api = new VisionAPI();
export default api;
