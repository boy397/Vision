**SARVAM GLASSES HACKATHON**

**Assistive Vision + Voice System**

**Dual-Mode: Medical** ✦ **Retail**

_Latency-first architecture · Voice-commanded · Configurable streaming · H100-backed inference_

# **1\. Concept Overview**

A wearable assistive system built for Sarvam's smart glasses that helps visually and audio-impaired users navigate two high-impact niches: the medical world (medicines, prescriptions) and the retail world (products, pricing, currency). The system runs in two modes - triggered on voice command, or continuous streaming when configured - and communicates everything through natural speech.

| **Target Users** | Visually impaired individuals navigating pharmacies, hospitals, shops, and markets |
| --- | --- |
| **Platform** | Sarvam Smart Glasses (camera + mic + speaker) |
| **Inference** | H100 GPU (300 GB RAM) for heavy models - local fallback for latency-critical steps |
| **Core Modes** | Voice-triggered (on demand) + Continuous streaming (configurable) |
| **Languages** | Sarvam STT/TTS for Indian language support - multilingual from day 1 |

# **2\. System Architecture**

The pipeline is split into two tiers to minimize latency. Tier 1 runs on-device (or edge) for fast, always-on detection. Tier 2 offloads to H100 only when deep understanding is needed. This avoids the 3-hop latency trap of sending every frame to a cloud model.

## **2.1 Pipeline - Tier 1 (On-Device / Edge, <100ms)**

| **Step** | **Component** | **Detail** | **Target Latency** |
| --- | --- | --- | --- |
| **S1** | **Camera Input** | Glasses camera captures frames. In continuous mode: 10-15 fps. In triggered mode: single capture on voice command. | **~0ms** |
| **S2** | **YOLO (on-device)** | Lightweight YOLO (YOLOv8n or YOLOv8s quantized) runs on-device. Detects object classes: pill bottle, blister pack, currency note, product box, prescription sheet. Acts as a gate - only fires Tier 2 when a relevant object is detected or state changes. | **20-40ms** |
| **S3** | **State Change Check** | Frame diff + class comparison. If detected objects are same as previous frame, skip Tier 2 call entirely. This is the primary latency optimization - prevents redundant API calls. | **~5ms** |
| **S4** | **STT - Sarvam / Google Speech** | Always-listening VAD (Voice Activity Detection). Triggers mode change or one-shot query. Routes voice commands to control layer. | **100-200ms** |

## **2.2 Pipeline - Tier 2 (H100 Inference, triggered)**

| **Step** | **Component** | **Detail** | **Target Latency** |
| --- | --- | --- | --- |
| **S5** | **Gemini Vision** | Receives cropped ROI (region of interest) from YOLO bounding box - not full frame. Analyzes object with mode-specific prompt (Med or Retail). Returns structured JSON. | **300-600ms** |
| **S6** | **Custom Model (H100)** | Optional: fine-tuned OCR/classification model for text on medicine packaging or currency. Runs alongside Gemini Vision in parallel. Use ensemble for confidence boosting. | **50-150ms** |
| **S7** | **Gemini Chat (STT follow-up)** | If user asks a follow-up voice question (e.g. 'is this safe with diabetes?'), sends context + question to Gemini chat. Returns conversational answer. | **400-800ms** |
| **S8** | **TTS - ElevenLabs / Sarvam** | Converts structured output to natural speech. ElevenLabs for high-quality English. Sarvam TTS for Hindi/regional. Response streamed - first audio chunk plays before full response is ready. | **100-300ms** |

## **2.3 Control Layer - Voice Commands**

The voice command layer is always-on and intercepts specific intents before routing to the vision pipeline. Commands are processed by a lightweight intent classifier running locally.

| **Voice Command** | **Action** | **Mode** |
| --- | --- | --- |
| **"Scan this"** | One-shot capture + full Tier 2 analysis | Triggered |
| **"What is this?"** | Same as scan - user-friendly alias | Triggered |
| **"Start watching"** | Enables continuous mode at configured fps | Continuous |
| **"Stop watching"** | Disables continuous mode, returns to idle | Off |
| **"Switch to medicine mode"** | Sets context to Med - changes YOLO classes + prompts | Config |
| **"Switch to retail mode"** | Sets context to Retail - changes YOLO classes + prompts | Config |
| **"Tell me more"** | Fires Gemini chat with last detection as context | Follow-up |
| **"Repeat that"** | Replays last TTS output | Utility |

# **3\. Niche Mode Details**

## **3.1 Medical Mode**

Triggered when YOLO detects: pill bottle, blister pack, medicine box, prescription paper. Gemini Vision prompt is specialized for medical context.

<div class="joplin-table-wrapper"><table><tbody><tr><th><p><strong>What the system extracts</strong></p></th></tr><tr><td><ul><li>Medicine name (brand + generic)</li></ul></td></tr><tr><td><ul><li>Expiry date - reads DD/MM/YYYY, MFG/EXP formats, handles faded ink</li></ul></td></tr><tr><td><ul><li>Dosage and usage instructions from packaging</li></ul></td></tr><tr><td><ul><li>Prescription reader - extracts doctor name, patient, drug, dose, frequency</li></ul></td></tr><tr><td><ul><li>Drug interaction warning flag (prompt-engineered - asks Gemini to flag if common allergen)</li></ul></td></tr><tr><td><ul><li>Manufacturer and batch number for recalls</li></ul></td></tr><tr><td></td></tr></tbody></table></div>

**Sample TTS output:**

_"This is Metformin 500mg by Sun Pharma. Expiry: March 2026. Dosage on pack says: take one tablet twice daily with meals. No known common allergens flagged."_

## **3.2 Retail Mode**

Triggered when YOLO detects: currency notes, product packaging, price tags, barcodes. Handles supermarkets, pharmacies, street vendors.

<div class="joplin-table-wrapper"><table><tbody><tr><th><p><strong>What the system extracts</strong></p></th></tr><tr><td><ul><li>Currency recognition - Indian notes (₹10 to ₹2000), reads denomination clearly</li></ul></td></tr><tr><td><ul><li>Product name and brand from packaging</li></ul></td></tr><tr><td><ul><li>Expiry date / best before date</li></ul></td></tr><tr><td><ul><li>Quantity / weight / volume from label</li></ul></td></tr><tr><td><ul><li>Price from price tag or sticker (OCR)</li></ul></td></tr><tr><td><ul><li>Barcode scan → product lookup (optional, if network available)</li></ul></td></tr><tr><td></td></tr></tbody></table></div>

**Sample TTS output:**

_"I can see a five hundred rupee note. Also on the table: Parle-G biscuits, 200 grams, best before August 2025, price tag shows 30 rupees."_

# **4\. Latency Strategy**

Latency is the single biggest risk for a glasses-based assistive product. A blind user cannot wait 3 seconds for a response. Target: first audio output within 800ms of trigger in triggered mode, under 1.5s end-to-end for complex queries.

<div class="joplin-table-wrapper"><table><tbody><tr><th><p><strong>Optimizations (prioritized by impact)</strong></p></th></tr><tr><td><ul><li>ROI cropping - send only the YOLO bounding box crop to Gemini, not the full frame. Reduces image tokens by ~70%, cuts API latency significantly.</li></ul></td></tr><tr><td><ul><li>State change gating - compare YOLO detections frame-to-frame. Skip Tier 2 if nothing changed. In continuous mode this eliminates ~80% of API calls.</li></ul></td></tr><tr><td><ul><li>Streaming TTS - start playing audio as soon as first sentence of response is ready. Don't wait for full response.</li></ul></td></tr><tr><td><ul><li>Parallel execution - run Gemini Vision and custom H100 model concurrently, merge results. Don't chain them sequentially.</li></ul></td></tr><tr><td><ul><li>Prompt caching - Gemini supports prompt caching. Cache the large system prompt (mode-specific instructions). Only variable part is the image.</li></ul></td></tr><tr><td><ul><li>Quantized YOLO on-device - use INT8 quantized YOLOv8n. Runs at 30+ fps on edge hardware, under 40ms per frame.</li></ul></td></tr><tr><td><ul><li>Local intent classifier - voice commands go through a local lightweight model (e.g. Whisper tiny + keyword match), not full STT for control commands. Reduces STT latency for common commands.</li></ul></td></tr><tr><td><ul><li>H100 warm model - keep Gemini-equivalent model loaded in VRAM. No cold start penalty.</li></ul></td></tr><tr><td></td></tr></tbody></table></div>

**Latency Budget (Triggered Mode)**

| **Step** | **Component** | **Detail** | **Target Latency** |
| --- | --- | --- | --- |
| **T1** | **Voice detection (VAD)** | Local VAD detects end of speech | **~50ms** |
| **T2** | **Intent classification** | Local keyword model classifies command | **~30ms** |
| **T3** | **Camera capture** | Single frame grab | **~16ms** |
| **T4** | **YOLO inference** | On-device YOLOv8n INT8 | **~40ms** |
| **T5** | **ROI crop + API call** | Crop + Gemini Vision with cached prompt | **~400ms** |
| **T6** | **TTS first chunk** | ElevenLabs/Sarvam streaming - first audio byte | **~150ms** |
| **TOTAL** | **First audio output** | User hears first word of response | **~700ms** |

# **5\. Model Selection**

Configurable at runtime. The system supports swapping models per mode without restarting.

| **Role** | **Primary Choice** | **Fallback** | **Reason** |
| --- | --- | --- | --- |
| Object Detection | YOLOv8n (INT8) | YOLOv8s | Fastest on edge, proven on packaging/currency |
| Scene Understanding | Gemini 1.5 Flash | Gemini 1.5 Pro | Flash: low latency. Pro: complex prescriptions |
| Custom OCR/Class. | Fine-tuned ViT on H100 | PaddleOCR | Fine-tune on Indian medicine/currency datasets |
| STT | Sarvam STT (primary) | Google Speech V2 | Sarvam for Indian languages, multilingual from day 1 |
| TTS | ElevenLabs (streaming) | Sarvam TTS | ElevenLabs quality, Sarvam for regional languages |
| Chat / Follow-up | Gemini 1.5 Flash | Gemini 1.5 Pro | Conversational follow-ups on last detection |
| Intent Classification | Whisper Tiny + rules | Sarvam STT | Low-latency local command parsing |

# **6\. Build Plan - Sprint Breakdown**

Designed to ship a demo-ready build as fast as possible. Focus on demo-ability over completeness - one clean end-to-end flow beats ten half-built features.

| **Sprint** | **Days** | **Goals** | **Deliverable** |
| --- | --- | --- | --- |
| **Sprint 1** | Day 1-2 | Environment setup. YOLO on-device running. Basic camera feed working on glasses (or laptop sim). Gemini Vision API connected with Med mode prompt. First end-to-end test: scan medicine box → get JSON → speak result. | **Working Med scan** |
| **Sprint 2** | Day 3-4 | Retail mode prompts + YOLO classes. Currency detection tested with Indian notes. Voice command layer: VAD + 'scan this' trigger. TTS streaming (ElevenLabs) connected. Sarvam STT for command input. | **Med + Retail triggered** |
| **Sprint 3** | Day 5-6 | State change gating implemented. ROI cropping tested - latency benchmarked. Continuous mode with configurable fps. Model swap config (Flash vs Pro toggle). H100 custom model endpoint up. | **Continuous mode + latency tuned** |
| **Sprint 4** | Day 7-8 | Follow-up chat ("tell me more") via Gemini. Repeat / stop commands. Regional language TTS via Sarvam. Polish TTS outputs - natural phrasing, not raw JSON read-out. Error handling (no object detected → "I don't see anything relevant"). | **Full voice UX polished** |
| **Sprint 5** | Day 9-10 | Demo script rehearsed. Two demo scenarios: pharmacist handing wrong medicine to blind user (Med) + blind shopper at counter (Retail). Latency logged and shown in demo. Backup modes if glasses fail (laptop + webcam). | **Demo-ready build** |

# **7\. Demo Strategy**

Hackathon judges remember one thing. Make it visceral - put judges in the user's shoes, not in a technical walkthrough.

<div class="joplin-table-wrapper"><table><tbody><tr><th><p><strong>Demo Script - Medical</strong></p></th></tr><tr><td><ul><li>Blindfold a volunteer (or have judge close eyes). Hand them a medicine bottle.</li></ul></td></tr><tr><td><ul><li>They say: "Scan this"</li></ul></td></tr><tr><td><ul><li>System says: "This is Paracetamol 650mg by Cipla. Expiry: November 2025. Dosage: one tablet every 6 hours as needed."</li></ul></td></tr><tr><td><ul><li>They say: "Tell me more" - system answers a follow-up about side effects.</li></ul></td></tr><tr><td><ul><li>Emotional hook: show what happens without this tool - reading tiny text, asking strangers.</li></ul></td></tr><tr><td></td></tr></tbody></table></div>

<div class="joplin-table-wrapper"><table><tbody><tr><th><p><strong>Demo Script - Retail</strong></p></th></tr><tr><td><ul><li>Volunteer reaches into bag of mixed currency notes.</li></ul></td></tr><tr><td><ul><li>They say: "What do I have?"</li></ul></td></tr><tr><td><ul><li>System says: "I can see a one hundred rupee note and a fifty rupee note. Total: one hundred fifty rupees."</li></ul></td></tr><tr><td><ul><li>Then scan a product: "Amul Butter, 100 grams, best before June 2025, price 55 rupees."</li></ul></td></tr><tr><td><ul><li>Emotional hook: independence at a shop counter without needing help from others.</li></ul></td></tr><tr><td></td></tr></tbody></table></div>

# **8\. Risks & Mitigations**

| **Risk** | **Severity** | **Mitigation** |
| --- | --- | --- |
| Gemini Vision latency spikes | **HIGH** | Fallback to H100 custom model. Cache prompts. Always benchmark both. |
| YOLO misses small text objects | **MED** | Use higher-res crop region. Train YOLO on medicine/currency dataset if time allows. |
| Glasses hardware constraints | **HIGH** | Have laptop + webcam as fallback. Same software stack, different camera input. |
| STT fails in noisy hackathon env. | **MED** | Test with EarPods mic. Add push-to-talk button fallback. Noise-cancelling VAD. |
| Currency OCR accuracy on worn notes | **MED** | Use ensemble: YOLO class + Gemini Vision + texture features. Confidence threshold before speaking. |
| Indian language TTS quality | **LOW** | Sarvam TTS prioritized. ElevenLabs as English fallback. Test voices early. |

# **9\. Tech Stack Summary**

| **Camera** | Sarvam Glasses camera → OpenCV frame capture |
| --- | --- |
| **Object Detection** | YOLOv8n (ultralytics) - INT8 quantized for edge |
| **Vision AI** | Gemini 1.5 Flash (primary), H100 fine-tuned ViT (parallel) |
| **STT** | Sarvam STT API + Whisper Tiny (local, for commands) |
| **TTS** | ElevenLabs streaming API + Sarvam TTS (regional) |
| **Voice Control** | Python VAD (webrtcvad) + intent rule engine |
| **Backend** | FastAPI (Python) - thin orchestration layer |
| **Config** | YAML config file - mode, fps, model selection, language |
| **Inference Server** | H100 via vLLM or TGI - custom model endpoint |
| **Languages** | Python (backend + ML), optional React web monitor UI |

**_Ship faster than anyone. Two clean demos. One visceral moment._**

_Focus: latency + demo emotion. Everything else is secondary._