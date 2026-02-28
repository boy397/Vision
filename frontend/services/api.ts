/**
 * API configuration for connecting to the Vision backend.
 * Change API_BASE to your backend server's address.
 */

// Dev tunnel URL for the backend
export const API_BASE = "https://l62bjdmv-8000.inc1.devtunnels.ms";
export const WS_URL = "wss://l62bjdmv-8000.inc1.devtunnels.ms/stream";

export type Mode = "medical" | "retail";

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

  private async safeFetch(url: string, init?: RequestInit): Promise<Response> {
    const res = await fetch(url, init);
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status}: ${text.slice(0, 200)}`);
    }
    return res;
  }

  async health(): Promise<HealthStatus> {
    const res = await this.safeFetch(`${this.base}/health`);
    return res.json();
  }

  async getConfig(): Promise<Record<string, any>> {
    const res = await this.safeFetch(`${this.base}/config`);
    return res.json();
  }

  async switchMode(
    mode: Mode,
  ): Promise<{ status: string; mode: string; message: string }> {
    const res = await this.safeFetch(`${this.base}/config/mode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    return res.json();
  }

  async scan(imageBase64: string): Promise<ScanResult> {
    const formData = new FormData();
    formData.append("image", {
      uri: imageBase64,
      type: "image/jpeg",
      name: "capture.jpg",
    } as any);

    const res = await this.safeFetch(`${this.base}/scan`, {
      method: "POST",
      body: formData,
    });
    return res.json();
  }

  async scanFromUri(imageUri: string): Promise<ScanResult> {
    const formData = new FormData();
    formData.append("image", {
      uri: imageUri,
      type: "image/jpeg",
      name: "capture.jpg",
    } as any);

    const res = await this.safeFetch(`${this.base}/scan`, {
      method: "POST",
      body: formData,
    });
    return res.json();
  }

  /**
   * Scan image and get back streaming TTS audio (MP3).
   * Returns the raw Response so the caller can stream the audio.
   */
  async scanWithAudio(imageUri: string): Promise<Response> {
    const formData = new FormData();
    formData.append("image", {
      uri: imageUri,
      type: "image/jpeg",
      name: "capture.jpg",
    } as any);

    const res = await this.safeFetch(`${this.base}/scan/audio`, {
      method: "POST",
      body: formData,
    });
    return res;
  }

  async sendVoice(audioUri: string): Promise<VoiceResult> {
    const formData = new FormData();
    formData.append("audio", {
      uri: audioUri,
      type: "audio/wav",
      name: "voice.wav",
    } as any);

    const res = await this.safeFetch(`${this.base}/voice`, {
      method: "POST",
      body: formData,
    });
    return res.json();
  }

  async chat(question: string): Promise<VoiceResult> {
    const res = await this.safeFetch(`${this.base}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    return res.json();
  }

  /**
   * Update the TTS provider on the backend (elevenlabs | sarvam).
   */
  async updateTtsProvider(provider: string): Promise<any> {
    const res = await this.safeFetch(`${this.base}/config/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider }),
    });
    return res.json();
  }

  /**
   * Convert text to speech audio (lightweight, no scan).
   * Returns raw Response with streaming MP3 audio.
   */
  async speakText(text: string): Promise<Response> {
    const res = await this.safeFetch(`${this.base}/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    return res;
  }
}

export const api = new VisionAPI();
export default api;
