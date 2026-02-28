/**
 * App-wide always-on voice command service.
 *
 * Uses a rapid record-cycle approach:
 *   1. Record a short ~2s audio clip
 *   2. Client-side VAD filters out silence/noise
 *   3. Send speech clips to backend /voice/listen endpoint
 *   4. Backend does STT + intent classification
 *   5. If a command is detected, emit it via scanEvents
 *   6. Immediately start the next recording cycle
 *
 * Key improvements over v1:
 *  - Rolling adaptive noise floor (recalibrates every ~30 cycles)
 *  - Percentile-based floor estimation (immune to transient spikes)
 *  - TTS mute window — suppresses VAD while the app is speaking
 *  - Threshold is computed relative to the ACTUAL noise floor, not a fixed -25dB clamp
 */

import { Audio } from "expo-av";
import scanEvents from "./scanEvents";
import { API_BASE } from "./api";

// ─── Tunables ────────────────────────────────────────────────────────────────

/** Duration of each listen cycle (ms). */
const LISTEN_CYCLE_MS = 2000;

/** Duration of the calibration cycle (ms). Longer = more stable noise floor. */
const CALIBRATION_CYCLE_MS = 3000;

/** Minimum clip length before we bother sending to the backend (ms). */
const MIN_AUDIO_MS = 400;

/**
 * How many dB above the noise floor counts as "speech".
 * A larger margin means fewer false positives but may miss soft voices.
 */
const SPEECH_MARGIN_DB = 15;

/**
 * Absolute minimum threshold regardless of noise floor.
 * Prevents triggering on truly inaudible sounds in a very quiet room.
 * e.g. noise floor -49 dB → threshold = -49 + 15 = -34 dB (not clamped to -25)
 */
const ABSOLUTE_MIN_THRESHOLD_DB = -40;

/**
 * Absolute maximum threshold.
 * Prevents the bar being set so high that normal speech is ignored in a loud room.
 */
const ABSOLUTE_MAX_THRESHOLD_DB = -10;

/**
 * Recalibrate (run a silent calibration cycle) every N active cycles.
 * Keeps the noise floor fresh as environment changes.
 */
const RECALIBRATE_EVERY_N_CYCLES = 40;

/**
 * Percentile of metering samples used to estimate the noise floor.
 * 85th percentile of a SILENT cycle is a robust noise floor estimate —
 * it catches occasional transient bumps without being thrown off by outliers.
 */
const NOISE_FLOOR_PERCENTILE = 0.85;

/**
 * After TTS finishes, suppress VAD for this many ms to let the mic settle
 * and avoid the app hearing its own voice.
 */
const TTS_MUTE_TAIL_MS = 1200;

// ─── Recording options ────────────────────────────────────────────────────────

const LISTEN_RECORDING_OPTIONS: Audio.RecordingOptions = {
    isMeteringEnabled: true,
    android: {
        extension: ".m4a",
        outputFormat: 2,  // MPEG_4
        audioEncoder: 3,  // AAC
        sampleRate: 16000,
        numberOfChannels: 1,
        bitRate: 64000,
    },
    ios: {
        extension: ".m4a",
        outputFormat: "applelossless" as any,
        audioQuality: 96,
        sampleRate: 16000,
        numberOfChannels: 1,
        bitRate: 64000,
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

// ─── Helpers ──────────────────────────────────────────────────────────────────

function percentile(sortedArr: number[], p: number): number {
    if (sortedArr.length === 0) return -160;
    const idx = Math.floor(sortedArr.length * p);
    return sortedArr[Math.min(idx, sortedArr.length - 1)];
}

// ─── Service ──────────────────────────────────────────────────────────────────

class VoiceStreamService {
    private running = false;
    private currentRecording: Audio.Recording | null = null;
    private cycleTimer: ReturnType<typeof setTimeout> | null = null;
    private statusListeners = new Set<StatusListener>();
    private _status: VoiceStreamStatus = "idle";
    private permissionGranted = false;

    // Cycle bookkeeping
    private cycleCount = 0;
    private isCalibrating = false;

    // Noise floor — starts pessimistically high so we don't fire before calibration
    private noiseFloor = -30;
    private noiseFloorReady = false;

    // Per-cycle metering
    private meteringSamples: number[] = [];

    // TTS mute window — set this timestamp when TTS starts/ends
    private ttsMuteUntil = 0;

    // ── Public API ────────────────────────────────────────────────────────────

    get status() { return this._status; }
    get isListening() { return this.running; }

    onStatus(listener: StatusListener) {
        this.statusListeners.add(listener);
        listener(this._status);
        return () => this.statusListeners.delete(listener);
    }

    /**
     * Call this right before the app plays TTS audio.
     * VAD will be suppressed until `ttsMuteUntil` expires.
     */
    notifyTtsStart(estimatedDurationMs: number) {
        this.ttsMuteUntil = Date.now() + estimatedDurationMs + TTS_MUTE_TAIL_MS;
        console.log(
            `[VoiceStream] 🔇 TTS mute active for ~${Math.round((estimatedDurationMs + TTS_MUTE_TAIL_MS) / 1000)}s`
        );
    }

    /**
     * Call this when TTS audio finishes playing (use the tail as extra buffer).
     */
    notifyTtsEnd() {
        this.ttsMuteUntil = Date.now() + TTS_MUTE_TAIL_MS;
        console.log(`[VoiceStream] 🔇 TTS finished — mute tail ${TTS_MUTE_TAIL_MS}ms`);
    }

    /** Start always-on voice listening. */
    async start() {
        if (this.running) {
            console.log("[VoiceStream] Already running, ignoring start()");
            return;
        }

        console.log("[VoiceStream] Starting always-on voice listening...");

        if (!this.permissionGranted) {
            const { granted } = await Audio.requestPermissionsAsync();
            if (!granted) {
                console.warn("[VoiceStream] Microphone permission denied");
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
        this.cycleCount = 0;
        this.noiseFloorReady = false;
        this.isCalibrating = true;

        this.setStatus("listening");
        this.startCycle();
    }

    /** Stop voice listening completely. */
    async stop() {
        console.log("[VoiceStream] Stopping voice listening...");
        this.running = false;

        if (this.cycleTimer) {
            clearTimeout(this.cycleTimer);
            this.cycleTimer = null;
        }

        if (this.currentRecording) {
            try { await this.currentRecording.stopAndUnloadAsync(); } catch { /* ignore */ }
            this.currentRecording = null;
        }

        this.setStatus("idle");
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    private setStatus(status: VoiceStreamStatus, detail?: string) {
        this._status = status;
        this.statusListeners.forEach((fn) => fn(status, detail));
    }

    private async startCycle() {
        if (!this.running) return;

        try {
            if (this.currentRecording) {
                try { await this.currentRecording.stopAndUnloadAsync(); } catch { /* ignore */ }
                this.currentRecording = null;
            }

            this.meteringSamples = [];

            const { recording } = await Audio.Recording.createAsync(LISTEN_RECORDING_OPTIONS);
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
                console.log(`[VoiceStream] 🎧 Calibrating noise floor — please be quiet...`);
            } else if (this.cycleCount % 10 === 1) {
                console.log(
                    `[VoiceStream] Cycle #${this.cycleCount} | Noise floor: ${this.noiseFloor.toFixed(1)} dB | ` +
                    `Threshold: ${this.computeThreshold().toFixed(1)} dB`
                );
            }

            this.cycleTimer = setTimeout(() => this.endCycleAndSend(), duration);
        } catch (err) {
            console.error("[VoiceStream] Error starting recording cycle:", err);
            if (this.running) {
                this.cycleTimer = setTimeout(() => this.startCycle(), 1000);
            }
        }
    }

    private async endCycleAndSend() {
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
        } catch (err) {
            console.warn("[VoiceStream] Error stopping recording:", err);
            this.currentRecording = null;
        }

        // ── Calibration cycle ────────────────────────────────────────────────
        if (this.isCalibrating) {
            this.updateNoiseFloor(samples);
            this.isCalibrating = false;
            this.noiseFloorReady = true;
            console.log(
                `[VoiceStream] ✅ Calibration done — noise floor: ${this.noiseFloor.toFixed(1)} dB, ` +
                `speech threshold: ${this.computeThreshold().toFixed(1)} dB`
            );
            if (this.running) this.startCycle();
            return;
        }

        // Kick off the next cycle immediately (fire-and-forget the send below)
        if (this.running) this.startCycle();

        // ── Periodic recalibration ────────────────────────────────────────────
        if (this.cycleCount % RECALIBRATE_EVERY_N_CYCLES === 0) {
            console.log(`[VoiceStream] 🔄 Triggering periodic recalibration...`);
            this.isCalibrating = true;
            // (the NEXT cycle will be a calibration cycle)
        }

        // ── TTS mute window ───────────────────────────────────────────────────
        if (Date.now() < this.ttsMuteUntil) {
            console.log(`[VoiceStream] 🔇 Dropping clip — TTS mute active`);
            return;
        }

        // ── Client-side VAD ───────────────────────────────────────────────────
        if (samples.length === 0) return;

        const maxDb = Math.max(...samples);
        const threshold = this.computeThreshold();

        if (maxDb < threshold) {
            if (this.cycleCount % 5 === 0) {
                console.log(
                    `[VoiceStream] 🤫 Quiet clip skipped (max: ${maxDb.toFixed(1)} dB < threshold: ${threshold.toFixed(1)} dB)`
                );
            }
            return;
        }

        console.log(
            `[VoiceStream] 🗣️ Speech detected (max: ${maxDb.toFixed(1)} dB > threshold: ${threshold.toFixed(1)} dB)`
        );

        // ── Send to backend ───────────────────────────────────────────────────
        if (uri && durationMs >= MIN_AUDIO_MS) {
            this.sendToBackend(uri, durationMs);
        }
    }

    /**
     * Compute the speech detection threshold from the current noise floor.
     * Clamped to [ABSOLUTE_MIN, ABSOLUTE_MAX] to stay sane in extreme environments.
     */
    private computeThreshold(): number {
        const raw = this.noiseFloor + SPEECH_MARGIN_DB;
        return Math.min(
            ABSOLUTE_MAX_THRESHOLD_DB,
            Math.max(ABSOLUTE_MIN_THRESHOLD_DB, raw)
        );
    }

    /**
     * Update the noise floor estimate from a set of metering samples.
     * Uses a high percentile of the sorted samples so occasional door-slams
     * or mic pops don't push the floor up unrealistically.
     */
    private updateNoiseFloor(samples: number[]) {
        if (samples.length === 0) return;
        const sorted = [...samples].sort((a, b) => a - b);
        const estimate = percentile(sorted, NOISE_FLOOR_PERCENTILE);

        if (!this.noiseFloorReady) {
            // First calibration — take the estimate directly
            this.noiseFloor = estimate;
        } else {
            // Subsequent recalibrations — blend with previous value (EMA α=0.4)
            // so a single noisy recalibration doesn't wildly shift the threshold
            this.noiseFloor = 0.6 * this.noiseFloor + 0.4 * estimate;
        }
    }

    /**
     * Send recorded audio to the backend /voice/listen endpoint.
     * Fire-and-forget — does not block the next recording cycle.
     */
    private async sendToBackend(uri: string, durationMs: number) {
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

            if (!response.ok) {
                const text = await response.text().catch(() => "");
                if (response.status !== 422) {
                    // 422 = no speech, expected — don't spam the log
                    console.warn(`[VoiceStream] Backend ${response.status}: ${text.slice(0, 200)}`);
                }
                return;
            }

            const data = await response.json();
            if (!data.has_command) return;

            console.log(
                `[VoiceStream] 🎙️ Command: action=${data.action}, text="${data.text ?? ""}"`
            );
            this.dispatchCommand(data);
        } catch (err) {
            if (this.cycleCount % 20 === 0) {
                console.warn("[VoiceStream] Send error (periodic log):", err);
            }
        }
    }

    /** Route a detected command to the appropriate event. */
    private dispatchCommand(data: any) {
        switch (data.action) {
            case "scan":
                console.log("[VoiceStream] → SCAN");
                scanEvents.emit("scan");
                break;
            case "start_continuous":
                console.log("[VoiceStream] → CONTINUOUS SCAN");
                scanEvents.emit("scan_continue");
                break;
            case "stop_continuous":
                console.log("[VoiceStream] → STOP");
                scanEvents.emit("scan_stop");
                break;
            default:
                console.log(`[VoiceStream] → Unhandled action: ${data.action}`);
        }
    }
}

// Singleton — shared across the entire app
export const voiceStream = new VoiceStreamService();
export default voiceStream;