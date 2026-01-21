import os
import sys
import wave
import threading
import numpy as np
import pyaudio
from faster_whisper import WhisperModel

# --- CONFIGURAÇÕES ---
MODEL_SIZE = "medium"  # 'small' ou 'medium' (small é mais rápido para 4GB VRAM)
DEVICE = "cuda"       # Usa a RTX 3050
COMPUTE_TYPE = "float16" # Otimizado para GPU
FILE_PT = "transcricao_pt.txt"
FILE_EN = "transcricao_en.txt"
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000 # Taxa recomendada pelo Whisper

class RealTimeTranslator:
    def __init__(self):
        print(f"Carregando modelo '{MODEL_SIZE}' na GPU...")
        self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        self.is_recording = False
        self.audio_buffer = []

    def transcribe_and_translate(self, audio_data):
        # Transcrição (Original)
        segments_pt, _ = self.model.transcribe(audio_data, language="pt", beam_size=5)
        text_pt = "".join([s.text for s in segments_pt]).strip()

        # Tradução (Direto para Inglês)
        segments_en, _ = self.model.transcribe(audio_data, task="translate", beam_size=5)
        text_en = "".join([s.text for s in segments_en]).strip()

        return text_pt, text_en

    def start(self):
        self.is_recording = True
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)
        
        print("\n[OK] Gravando... Pressione CTRL+C para parar.\n")
        
        try:
            with open(FILE_PT, "a", encoding="utf-8") as f_pt, open(FILE_EN, "a", encoding="utf-8") as f_en:
                while self.is_recording:
                    # Captura um bloco de 3-5 segundos para processamento estável
                    frames = []
                    for _ in range(0, int(RATE / CHUNK_SIZE * 3)): 
                        data = stream.read(CHUNK_SIZE)
                        frames.append(data)
                    
                    # Converte para formato aceito pelo modelo
                    audio_np = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
                    
                    pt, en = self.transcribe_and_translate(audio_np)
                    
                    if pt:
                        print(f"PT: {pt}")
                        print(f"EN: {en}")
                        print("-" * 20)
                        f_pt.write(pt + "\n")
                        f_en.write(en + "\n")
                        f_pt.flush()
                        f_en.flush()

        except KeyboardInterrupt:
            print("\nFinalizando...")
        finally:
            self.is_recording = False
            stream.stop_stream()
            stream.close()
            p.terminate()

    def transcribe_file(self, file_path):
        print(f"Processando arquivo: {file_path}")
        segments, _ = self.model.transcribe(file_path, language="pt")
        with open("arquivo_transcrito.txt", "w", encoding="utf-8") as f:
            for s in segments:
                print(s.text)
                f.write(s.text + "\n")

if __name__ == "__main__":
    app = RealTimeTranslator()
    # Para transcrever arquivo, use: app.transcribe_file("audio.wav")
    app.start()