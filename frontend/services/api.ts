/**
 * API configuration for connecting to the Vision backend.
 * Change API_BASE to your backend server's address.
 */

// Active ngrok tunnel — update this when ngrok URL changes
export const API_BASE = "https://fluent-supersecularly-jefferson.ngrok-free.dev";
export const WS_URL = "wss://fluent-supersecularly-jefferson.ngrok-free.dev/stream";
export const VOICE_WS_URL = "wss://fluent-supersecularly-jefferson.ngrok-free.dev/voice/stream";

export type Mode = "medical" | "retail";

export interface Detection {
  class_name: string;
  confidence: number;
  bbox: number[];
  track_id?: number; // Persistent tracking ID from BoT-SORT
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
  llm_model?: string;
  tts_provider: string;
  stt_provider: string;
  detection_enabled: boolean;
  detector_stats?: Record<string, any>;
}

export interface DebugStats {
  detector: Record<string, any>;
  pipeline: Record<string, any>;
  tts?: Record<string, any>;
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
    const start = Date.now();
    console.log(`[API] → ${init?.method || "GET"} ${url}`);

    // Required to bypass ngrok browser warning interstitial page
    init = {
      ...init,
      headers: {
        "ngrok-skip-browser-warning": "true",
        ...(init?.headers ?? {}),
      },
    };

    try {
      const res = await fetch(url, init);
      const elapsed = Date.now() - start;

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        console.error(
          `[API] ← ${res.status} ${url} (${elapsed}ms): ${text.slice(0, 200)}`,
        );
        throw new Error(`HTTP ${res.status}: ${text.slice(0, 200)}`);
      }

      console.log(`[API] ← ${res.status} ${url} (${elapsed}ms)`);
      return res;
    } catch (err) {
      const elapsed = Date.now() - start;
      console.error(`[API] ✖ ${url} (${elapsed}ms):`, err);
      throw err;
    }
  }

  async health(): Promise<HealthStatus> {
    const res = await this.safeFetch(`${this.base}/health`);
    return res.json();
  }

  async getConfig(): Promise<Record<string, any>> {
    const res = await this.safeFetch(`${this.base}/config`);
    return res.json();
  }

  /** Get debug stats from all pipeline components */
  async getDebugStats(): Promise<DebugStats> {
    const res = await this.safeFetch(`${this.base}/debug/stats`);
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
    const start = Date.now();
    console.log(`[API] scanFromUri: starting...`);

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

    const data = await res.json();
    const elapsed = Date.now() - start;

    // Log everything the backend returned
    console.log(
      `[API] scanFromUri: ${elapsed}ms, ` +
      `${data.detections?.length || 0} detections, ` +
      `state_changed=${data.state_changed}, ` +
      `tts_len=${data.tts_text?.length || 0}`,
    );
    console.log(`[API] scanFromUri LLM analysis:`, JSON.stringify(data.analysis || {}).substring(0, 500));
    console.log(`[API] scanFromUri tts_text: "${data.tts_text || '(empty)'}"`);
    if (data.mode) console.log(`[API] scanFromUri mode: ${data.mode}`);

    return data;
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
    const start = Date.now();
    console.log(`[API] sendVoice: starting... audioUri=${audioUri}`);

    const formData = new FormData();
    formData.append("audio", {
      uri: audioUri,
      type: "audio/mp4",  // Expo records as M4A/AAC on both Android and iOS
      name: "voice.m4a",
    } as any);

    const res = await this.safeFetch(`${this.base}/voice`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    const elapsed = Date.now() - start;
    console.log(
      `[API] sendVoice: ${elapsed}ms, action=${data.action}, text="${data.text || ""}"`,
    );
    console.log(`[API] sendVoice full response:`, JSON.stringify(data).substring(0, 500));

    return data;
  }

  async chat(question: string): Promise<VoiceResult> {
    const res = await this.safeFetch(`${this.base}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    return res.json();
  }

  /** Switch LLM vision provider at runtime (google | groq). */
  async switchLlmProvider(provider: string): Promise<any> {
    const res = await this.safeFetch(`${this.base}/config/llm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider }),
    });
    return res.json();
  }

  /**
   * Toggle YOLO detection on/off at runtime.
   * When disabled: frame → Gemini Vision directly (no YOLO gate).
   */
  async toggleDetection(enabled: boolean): Promise<any> {
    const res = await this.safeFetch(`${this.base}/config/detection`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
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
    const start = Date.now();
    console.log(`[API] speakText: text_len=${text.length}`);

    const res = await this.safeFetch(`${this.base}/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const elapsed = Date.now() - start;
    console.log(`[API] speakText: response in ${elapsed}ms`);
    return res;
  }
}

export const api = new VisionAPI();
export default api;
