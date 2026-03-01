/**
 * voiceStream.ts — Sarvam WebSocket Streaming Voice Service (v2)
 *
 * Architecture:
 *   Phone mic (WAV 16kHz) → /voice/sarvam WS → Sarvam STT WS (saaras:v3)
 *   → Intent classify → Vision LLM (if scan)
 *   → Sarvam TTS WS (bulbul:v3) → MP3 chunks → Phone speaker
 *
 * Key improvements over v1 (REST /voice/listen polling):
 *  - True streaming: first audio byte from TTS arrives ~200ms after speech ends
 *  - No polling loop — single persistent WebSocket handles everything
 *  - Sarvam saaras:v3 STT: much higher accuracy than saarika:v2.5
 *  - TTS audio arrives as MP3 chunks streamed directly over the same WS
 *  - Photo for scan is taken on frontend and sent as base64 over the WS
 *
 * Protocol (client → server):
 *   binary:               WAV audio chunk bytes (accumulates until flush)
 *   {"action":"flush"}    User stopped speaking — process accumulated audio
 *   {"action":"ping"}     Keepalive
 *   {"action":"scan_frame","frame_b64":"..."}  JPEG frame for vision scan
 *   {"action":"reset"}    Reset session state
 *
 * Protocol (server → client):
 *   {"type":"transcript","text":"..."}    Live STT result
 *   {"type":"intent","action":"..."}      Classified intent
 *   {"type":"scan_result",...}            Vision scan result
 *   binary:                               MP3 TTS audio chunks (play immediately)
 *   {"type":"tts_done"}                   End of TTS audio
 *   {"type":"pong"}                       Keepalive response
 */

import { Audio } from "expo-av";
import { CameraView } from "expo-camera";
import scanEvents from "./scanEvents";
import { SARVAM_WS_URL, API_BASE } from "./api";

// ─── Tunables ────────────────────────────────────────────────────────────────

/** How long each recording cycle lasts (ms). VAD decides whether to send. */
const LISTEN_CYCLE_MS = 1800;

/** Calibration cycle duration — measure noise floor once on start. */
const CALIBRATION_CYCLE_MS = 2500;

/** Speech must be above noise floor by this margin to trigger a flush. */
const SPEECH_MARGIN_DB = 14;

/** Absolute bounds for the computed speech threshold (dB). */
const ABSOLUTE_MIN_THRESHOLD_DB = -42;
const ABSOLUTE_MAX_THRESHOLD_DB = -10;

/** Re-calibrate every N active cycles. */
const RECALIBRATE_EVERY_N_CYCLES = 50;

/** Noise floor percentile (to exclude transient spikes during calibration). */
const NOISE_FLOOR_PERCENTILE = 0.85;

/** Suppress VAD for this long after TTS finishes (ms). */
const TTS_MUTE_TAIL_MS = 1200;

/** WebSocket reconnect: base & max delays (ms). */
const RECONNECT_BASE_MS = 1500;
const RECONNECT_MAX_MS = 20000;

/** Keepalive ping interval (ms). */
const PING_INTERVAL_MS = 20000;

/** Minimum recorded clip length to bother flushing (ms). */
const MIN_CLIP_MS = 300;

// ─── Recording options (WAV preferred for Sarvam streaming STT) ──────────────
// NOTE: Sarvam streaming STT only accepts WAV or raw PCM.
// Expo on Android records M4A/AAC natively, so we use WAV on iOS and M4A on Android.
// The backend handles format detection via magic bytes.

const RECORDING_OPTIONS: Audio.RecordingOptions = {
    isMeteringEnabled: true,
    android: {
        extension: ".m4a",
        outputFormat: 2,   // MPEG_4
        audioEncoder: 3,   // AAC
        sampleRate: 16000,
        numberOfChannels: 1,
        bitRate: 64000,
    },
    ios: {
        extension: ".wav",
        outputFormat: "lpcm" as any, // Linear PCM = WAV
        audioQuality: 96,            // MEDIUM
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

// ─── Types ────────────────────────────────────────────────────────────────────

export type VoiceStreamStatus = "idle" | "listening" | "processing" | "error";
type StatusListener = (status: VoiceStreamStatus, detail?: string) => void;

/** Camera ref injected by the scan screen so we can take photos on demand. */
let _cameraRef: React.RefObject<CameraView | null> | null = null;

export function registerCamera(ref: React.RefObject<CameraView | null>) {
    _cameraRef = ref;
}

export function unregisterCamera() {
    _cameraRef = null;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function percentile(sortedArr: number[], p: number): number {
    if (sortedArr.length === 0) return -160;
    const idx = Math.floor(sortedArr.length * p);
    return sortedArr[Math.min(idx, sortedArr.length - 1)];
}

/** Read a file URI and return Uint8Array bytes (React Native compatible). */
async function readFileBytes(uri: string): Promise<Uint8Array> {
    const res = await fetch(uri);
    const buf = await res.arrayBuffer();
    return new Uint8Array(buf);
}

// ─── Service ──────────────────────────────────────────────────────────────────

class VoiceStreamService {
    private running = false;
    private currentRecording: Audio.Recording | null = null;
    private cycleTimer: ReturnType<typeof setTimeout> | null = null;
    private pingTimer: ReturnType<typeof setInterval> | null = null;
    private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    private reconnectAttempt = 0;
    private intentionalClose = false;

    private ws: WebSocket | null = null;
    private wsReady = false;

    private statusListeners = new Set<StatusListener>();
    private _status: VoiceStreamStatus = "idle";
    private permissionGranted = false;

    // Cycle bookkeeping
    private cycleCount = 0;
    private isCalibrating = false;
    private noiseFloor = -30;
    private noiseFloorReady = false;
    private meteringSamples: number[] = [];

    // TTS mute window
    private ttsMuteUntil = 0;

    // Pending TTS audio chunks — collect until tts_done, then play
    private ttsBuffer: Uint8Array[] = [];
    private ttsPlaying = false;

    // ── Public API ────────────────────────────────────────────────────────────

    get status() { return this._status; }
    get isListening() { return this.running; }

    onStatus(listener: StatusListener) {
        this.statusListeners.add(listener);
        listener(this._status);
        return () => this.statusListeners.delete(listener);
    }

    notifyTtsStart(estimatedDurationMs: number) {
        this.ttsMuteUntil = Date.now() + estimatedDurationMs + TTS_MUTE_TAIL_MS;
    }

    notifyTtsEnd() {
        this.ttsMuteUntil = Date.now() + TTS_MUTE_TAIL_MS;
    }

    /** Start always-on voice listening via Sarvam WS pipeline. */
    async start() {
        if (this.running) return;
        console.log("[VoiceStream] 🚀 Starting Sarvam WebSocket voice pipeline...");

        if (!this.permissionGranted) {
            const { granted } = await Audio.requestPermissionsAsync();
            if (!granted) {
                this.setStatus("error", "Microphone permission denied");
                return;
            }
            this.permissionGranted = true;
        }

        await Audio.setAudioModeAsync({
            allowsRecordingIOS: true,
            playsInSilentModeIOS: true,
        });

        this.running = true;
        this.intentionalClose = false;
        this.cycleCount = 0;
        this.noiseFloorReady = false;
        this.isCalibrating = true;

        this.connectWS();
        this.setStatus("listening");
        this.startCycle();
    }

    /** Stop voice listening completely. */
    async stop() {
        console.log("[VoiceStream] ⏹ Stopping...");
        this.running = false;
        this.intentionalClose = true;

        this.clearTimers();

        if (this.currentRecording) {
            try { await this.currentRecording.stopAndUnloadAsync(); } catch { }
            this.currentRecording = null;
        }

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        this.setStatus("idle");
    }

    // ── WebSocket ─────────────────────────────────────────────────────────────

    private connectWS() {
        if (this.ws?.readyState === WebSocket.OPEN) return;

        console.log(`[VoiceStream] WS connecting to ${SARVAM_WS_URL}...`);
        const ws = new WebSocket(SARVAM_WS_URL);
        ws.binaryType = "arraybuffer";

        ws.onopen = () => {
            console.log("[VoiceStream] ✅ WS connected to /voice/sarvam");
            this.wsReady = true;
            this.reconnectAttempt = 0;
            this.startPing();
        };

        ws.onmessage = (event) => this.handleWSMessage(event);

        ws.onerror = (err) => {
            console.warn("[VoiceStream] WS error:", err);
            this.wsReady = false;
        };

        ws.onclose = (event) => {
            console.log(`[VoiceStream] WS closed (code=${event?.code})`);
            this.wsReady = false;
            this.stopPing();
            if (!this.intentionalClose && this.running) {
                this.scheduleReconnect();
            }
        };

        this.ws = ws;
    }

    private scheduleReconnect() {
        const delay = Math.min(
            RECONNECT_BASE_MS * Math.pow(1.5, this.reconnectAttempt),
            RECONNECT_MAX_MS,
        );
        this.reconnectAttempt++;
        console.log(`[VoiceStream] Reconnecting in ${(delay / 1000).toFixed(1)}s (attempt ${this.reconnectAttempt})`);
        this.reconnectTimer = setTimeout(() => this.connectWS(), delay);
    }

    private startPing() {
        this.stopPing();
        this.pingTimer = setInterval(() => {
            if (this.ws?.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ action: "ping" }));
            }
        }, PING_INTERVAL_MS);
    }

    private stopPing() {
        if (this.pingTimer) {
            clearInterval(this.pingTimer);
            this.pingTimer = null;
        }
    }

    private async handleWSMessage(event: MessageEvent) {
        // Binary = TTS audio chunk
        if (event.data instanceof ArrayBuffer) {
            this.ttsBuffer.push(new Uint8Array(event.data));
            return;
        }

        // Text = JSON control message
        try {
            const msg = JSON.parse(event.data as string);
            const type = msg.type || "";

            if (type === "tts_done") {
                // All TTS chunks arrived — play them
                if (this.ttsBuffer.length > 0) {
                    const combined = this._concatBuffers(this.ttsBuffer);
                    this.ttsBuffer = [];
                    await this._playMp3(combined);
                }
                return;
            }

            if (type === "transcript") {
                console.log(`[VoiceStream] 📝 Transcript: "${msg.text}"`);
                return;
            }

            if (type === "intent") {
                console.log(`[VoiceStream] 🎯 Intent: action=${msg.action}`);
                this.handleIntent(msg);
                return;
            }

            if (type === "scan_result") {
                console.log(`[VoiceStream] 📸 Scan result: ${msg.detections?.length || 0} dets`);
                // The scan result TTS comes as a separate TTS audio stream
                return;
            }

            if (type === "pong") {
                return; // keepalive OK
            }

            console.log("[VoiceStream] WS msg:", msg);
        } catch (e) {
            console.warn("[VoiceStream] WS parse error:", e);
        }
    }

    private handleIntent(msg: any) {
        switch (msg.action) {
            case "scan":
                // Server wants a photo — take it and send frame_b64
                scanEvents.emit("scan"); // Trigger UI feedback
                this._takeAndSendPhoto();
                break;
            case "start_continuous":
                scanEvents.emit("scan_continue");
                break;
            case "stop_continuous":
                scanEvents.emit("scan_stop");
                break;
            case "switch_mode":
                console.log(`[VoiceStream] Mode switch to: ${msg.mode}`);
                break;
            case "follow_up":
            case "repeat":
                // TTS response comes as binary audio — already handled in handleWSMessage
                break;
            default:
                console.log(`[VoiceStream] Unhandled intent: ${msg.action}`);
        }
    }

    /** Take a photo using the registered camera and send it to the backend. */
    private async _takeAndSendPhoto() {
        if (!_cameraRef?.current) {
            console.warn("[VoiceStream] No camera registered — cannot send scan frame");
            return;
        }
        try {
            const photo = await _cameraRef.current.takePictureAsync({
                quality: 0.7,
                base64: true,
            });
            if (!photo?.base64) {
                console.warn("[VoiceStream] Photo capture returned no base64");
                return;
            }
            if (this.ws?.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({
                    action: "scan_frame",
                    frame_b64: photo.base64,
                }));
                console.log(`[VoiceStream] 📸 Sent scan frame (${photo.base64.length} chars)`);
            }
        } catch (e) {
            console.error("[VoiceStream] Photo capture error:", e);
        }
    }

    /** Concatenate Uint8Array chunks into one. */
    private _concatBuffers(chunks: Uint8Array[]): Uint8Array {
        const total = chunks.reduce((acc, c) => acc + c.length, 0);
        const out = new Uint8Array(total);
        let offset = 0;
        for (const c of chunks) {
            out.set(c, offset);
            offset += c.length;
        }
        return out;
    }

    /** Play MP3 bytes using expo-av. */
    private async _playMp3(mp3Bytes: Uint8Array) {
        if (this.ttsPlaying) return; // Prevent overlap
        this.ttsPlaying = true;

        // Estimate duration for mute window (~150 bytes/ms at 128kbps)
        const estDurationMs = Math.max(500, mp3Bytes.length / 16);
        this.ttsMuteUntil = Date.now() + estDurationMs + TTS_MUTE_TAIL_MS;

        try {
            // Convert Uint8Array to base64
            let binary = "";
            const bytes = mp3Bytes;
            for (let i = 0; i < bytes.byteLength; i++) {
                binary += String.fromCharCode(bytes[i]);
            }
            const base64 = btoa(binary);

            const { sound } = await Audio.Sound.createAsync(
                { uri: `data:audio/mpeg;base64,${base64}` },
                { shouldPlay: true },
            );

            await new Promise<void>((resolve) => {
                sound.setOnPlaybackStatusUpdate((status) => {
                    if (status.isLoaded && status.didJustFinish) {
                        resolve();
                    }
                });
                // Fallback timeout in case callback doesn't fire
                setTimeout(resolve, estDurationMs + 3000);
            });

            await sound.unloadAsync().catch(() => { });
        } catch (e) {
            console.warn("[VoiceStream] TTS playback error:", e);
        } finally {
            this.ttsPlaying = false;
            this.ttsMuteUntil = Date.now() + TTS_MUTE_TAIL_MS;
        }
    }

    // ── Recording cycle (client-side VAD → WS flush) ─────────────────────────

    private async startCycle() {
        if (!this.running) return;

        try {
            if (this.currentRecording) {
                try { await this.currentRecording.stopAndUnloadAsync(); } catch { }
                this.currentRecording = null;
            }

            this.meteringSamples = [];
            const { recording } = await Audio.Recording.createAsync(RECORDING_OPTIONS);
            this.currentRecording = recording;
            this.cycleCount++;

            recording.setProgressUpdateInterval(100);
            recording.setOnRecordingStatusUpdate((status) => {
                if (status.metering !== undefined) {
                    this.meteringSamples.push(status.metering);
                }
            });

            const duration = this.isCalibrating ? CALIBRATION_CYCLE_MS : LISTEN_CYCLE_MS;

            if (this.isCalibrating) {
                console.log("[VoiceStream] 🎧 Calibrating noise floor...");
            }

            this.cycleTimer = setTimeout(() => this.endCycleAndFlush(), duration);
        } catch (err) {
            console.error("[VoiceStream] Cycle start error:", err);
            if (this.running) {
                this.cycleTimer = setTimeout(() => this.startCycle(), 1000);
            }
        }
    }

    private async endCycleAndFlush() {
        if (!this.running || !this.currentRecording) {
            if (this.running) this.startCycle();
            return;
        }

        let uri: string | null = null;
        let durationMs = 0;
        const samples = [...this.meteringSamples];

        try {
            const status = await this.currentRecording.getStatusAsync();
            durationMs = status.durationMillis || 0;
            await this.currentRecording.stopAndUnloadAsync();
            uri = this.currentRecording.getURI();
            this.currentRecording = null;
        } catch {
            this.currentRecording = null;
        }

        // ── Calibration cycle ──────────────────────────────────────────────────
        if (this.isCalibrating) {
            this.updateNoiseFloor(samples);
            this.isCalibrating = false;
            this.noiseFloorReady = true;
            console.log(
                `[VoiceStream] ✅ Calibrated: floor=${this.noiseFloor.toFixed(1)}dB, ` +
                `threshold=${this.computeThreshold().toFixed(1)}dB`
            );
            if (this.running) this.startCycle();
            return;
        }

        // Start next cycle immediately (fire-and-forget send)
        if (this.running) this.startCycle();

        // Periodic recalibration
        if (this.cycleCount % RECALIBRATE_EVERY_N_CYCLES === 0) {
            this.isCalibrating = true;
        }

        // TTS mute window — drop audio while speaking
        if (Date.now() < this.ttsMuteUntil) {
            return;
        }

        // Client-side VAD
        if (samples.length === 0) return;
        const maxDb = Math.max(...samples);
        const threshold = this.computeThreshold();

        if (maxDb < threshold) {
            if (this.cycleCount % 5 === 0) {
                console.log(`[VoiceStream] 🤫 Quiet (max=${maxDb.toFixed(1)}dB < ${threshold.toFixed(1)}dB)`);
            }
            return;
        }

        console.log(`[VoiceStream] 🗣️ Speech (max=${maxDb.toFixed(1)}dB > ${threshold.toFixed(1)}dB)`);

        // Send to backend via WebSocket
        if (uri && durationMs >= MIN_CLIP_MS) {
            await this.sendAudioViaWS(uri);
        }
    }

    /** Read the recorded file and send it as binary over the WS, then flush. */
    private async sendAudioViaWS(uri: string) {
        if (!this.ws || !this.wsReady) {
            console.warn("[VoiceStream] WS not ready — falling back to REST");
            await this.sendToRestFallback(uri);
            return;
        }

        try {
            const bytes = await readFileBytes(uri);
            this.ws.send(bytes.buffer as ArrayBuffer);
            this.ws.send(JSON.stringify({ action: "flush" }));
            console.log(`[VoiceStream] 📤 Sent ${bytes.length} bytes + flush to WS`);
        } catch (e) {
            console.error("[VoiceStream] WS send error:", e);
            // Try REST fallback
            await this.sendToRestFallback(uri);
        }
    }

    /** Fallback to the old REST /voice/listen endpoint when WS is unavailable. */
    private async sendToRestFallback(uri: string) {
        try {
            const formData = new FormData();
            formData.append("audio", {
                uri,
                type: "audio/mp4",
                name: "listen.m4a",
            } as any);

            const response = await fetch(`${API_BASE}/voice/listen`, {
                method: "POST",
                body: formData,
                headers: { "ngrok-skip-browser-warning": "true" },
            });

            if (!response.ok) return;

            const data = await response.json();
            if (!data.has_command) return;

            console.log(`[VoiceStream] 🎙️ REST fallback: action=${data.action}`);
            switch (data.action) {
                case "scan": scanEvents.emit("scan"); break;
                case "start_continuous": scanEvents.emit("scan_continue"); break;
                case "stop_continuous": scanEvents.emit("scan_stop"); break;
            }
        } catch (e) {
            console.warn("[VoiceStream] REST fallback error:", e);
        }
    }

    // ── VAD helpers ───────────────────────────────────────────────────────────

    private computeThreshold(): number {
        const raw = this.noiseFloor + SPEECH_MARGIN_DB;
        return Math.min(ABSOLUTE_MAX_THRESHOLD_DB, Math.max(ABSOLUTE_MIN_THRESHOLD_DB, raw));
    }

    private updateNoiseFloor(samples: number[]) {
        if (samples.length === 0) return;
        const sorted = [...samples].sort((a, b) => a - b);
        const estimate = percentile(sorted, NOISE_FLOOR_PERCENTILE);
        if (!this.noiseFloorReady) {
            this.noiseFloor = estimate;
        } else {
            this.noiseFloor = 0.6 * this.noiseFloor + 0.4 * estimate;
        }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    private setStatus(status: VoiceStreamStatus, detail?: string) {
        this._status = status;
        this.statusListeners.forEach((fn) => fn(status, detail));
    }

    private clearTimers() {
        if (this.cycleTimer) { clearTimeout(this.cycleTimer); this.cycleTimer = null; }
        if (this.pingTimer) { clearInterval(this.pingTimer); this.pingTimer = null; }
        if (this.reconnectTimer) { clearTimeout(this.reconnectTimer); this.reconnectTimer = null; }
    }
}

// Singleton
export const voiceStream = new VoiceStreamService();
export default voiceStream;