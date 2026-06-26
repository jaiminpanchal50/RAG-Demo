import torch
import scipy.io.wavfile as wav
import sounddevice as sd
import numpy as np
import tempfile
import os
from transformers import VitsModel, AutoTokenizer

# ── TTS Config ───────────────────────────────────────────────
TTS_MODELS = {
    "hi": "facebook/mms-tts-hin",   # Hindi
    "gu": "facebook/mms-tts-guj",   # Gujarati
    "en": "facebook/mms-tts-eng",   # English
}

SAMPLE_RATE = 16000
# ─────────────────────────────────────────────────────────────

class SvaraTTS:
    def __init__(self):
        self.models     = {}
        self.tokenizers = {}
        self._load_models()

    def _load_models(self):
        print("🔊 Loading TTS models...")
        for lang, model_id in TTS_MODELS.items():
            print(f"   Loading {lang} → {model_id}...")
            self.tokenizers[lang] = AutoTokenizer.from_pretrained(model_id)
            self.models[lang]     = VitsModel.from_pretrained(model_id)
            self.models[lang].eval()
        print("   ✅ All TTS models ready\n")

    def detect_language(self, text: str) -> str:
        """
        Simple language detector based on character ranges.
        Gujarati: \u0A80-\u0AFF
        Hindi (Devanagari): \u0900-\u097F
        Default: English
        """
        gujarati_chars = sum(1 for c in text if '\u0A80' <= c <= '\u0AFF')
        hindi_chars    = sum(1 for c in text if '\u0900' <= c <= '\u097F')

        if gujarati_chars > hindi_chars and gujarati_chars > 2:
            return "gu"
        elif hindi_chars > 2:
            return "hi"
        else:
            return "en"

    def speak(self, text: str, lang: str = None):
        """
        Convert text to speech and play on speakers.
        Auto-detects language if not specified.
        """
        if not text or not text.strip():
            return

        # Auto-detect language
        if lang is None:
            lang = self.detect_language(text)

        print(f"🔊 TTS [{lang.upper()}]: {text}")

        if lang not in self.models:
            print(f"   ⚠️  Language '{lang}' not supported, using English")
            lang = "en"

        try:
            tokenizer = self.tokenizers[lang]
            model     = self.models[lang]

            inputs = tokenizer(text, return_tensors="pt")

            with torch.no_grad():
                output = model(**inputs).waveform

            # Convert to numpy and play
            audio = output.squeeze().numpy()

            # Normalize audio
            audio = audio / np.max(np.abs(audio) + 1e-8)
            audio = (audio * 32767).astype(np.int16)

            # Play directly on speakers
            sd.play(audio, samplerate=model.config.sampling_rate)
            sd.wait()  # wait until audio finishes playing

            print("   ✅ Audio played\n")

        except Exception as e:
            print(f"   ⚠️  TTS error: {e}")

    def speak_and_save(self, text: str, output_path: str = "output.wav", lang: str = None):
        """
        Convert text to speech, play and save to file.
        """
        if not text or not text.strip():
            return

        if lang is None:
            lang = self.detect_language(text)

        if lang not in self.models:
            lang = "en"

        tokenizer = self.tokenizers[lang]
        model     = self.models[lang]

        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = model(**inputs).waveform

        audio = output.squeeze().numpy()

        # Save to file
        wav.write(output_path, rate=model.config.sampling_rate,
                  data=(audio * 32767).astype(np.int16))
        print(f"   💾 Saved to {output_path}")

        # Also play
        audio_int = (audio / np.max(np.abs(audio) + 1e-8) * 32767).astype(np.int16)
        sd.play(audio_int, samplerate=model.config.sampling_rate)
        sd.wait()