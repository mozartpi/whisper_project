import queue
import threading
import numpy as np
import pyaudio
from faster_whisper import WhisperModel
from scipy.signal import resample

# --- CONFIGURAÇÕES PRO ---
MODEL_SIZE = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16" # Melhor para sua RTX 3050 4GB

INPUT_RATE = 48000  # Captura em alta fidelidade (pode tentar 96000)
TARGET_RATE = 16000 # Taxa que o Whisper exige
SILENCE_THRESHOLD = 0.005 # Sensibilidade do fim de fala (ajuste se necessário)
MAX_BUFFER_SECONDS = 25   # Força o processamento se a fala for muito longa

audio_queue = queue.Queue()
is_running = True

def audio_capture_thread():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=INPUT_RATE, 
                    input=True, frames_per_buffer=2048)
    
    print(f"[OK] Capturando a {INPUT_RATE}Hz. Fale naturalmente...")
    
    current_phrase_buffer = []
    silence_counter = 0

    while is_running:
        data = stream.read(2048, exception_on_overflow=False)
        audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        current_phrase_buffer.append(audio_chunk)
        
        # Lógica simples de detecção de fim de frase (Volume baixo por X tempo)
        if np.abs(audio_chunk).mean() < SILENCE_THRESHOLD:
            silence_counter += 1
        else:
            silence_counter = 0

        # Se houver silêncio por ~1.5s ou o buffer estiver cheio, envia para processar
        buffer_len_sec = (len(current_phrase_buffer) * 2048) / INPUT_RATE
        if (silence_counter > 30 and buffer_len_sec > 2.0) or buffer_len_sec > MAX_BUFFER_SECONDS:
            full_audio = np.concatenate(current_phrase_buffer)
            
            # Resampling de alta qualidade para 16kHz
            num_samples = int(len(full_audio) * TARGET_RATE / INPUT_RATE)
            resampled_audio = resample(full_audio, num_samples)
            
            audio_queue.put(resampled_audio)
            current_phrase_buffer = []
            silence_counter = 0

    stream.stop_stream()
    stream.close()
    p.terminate()

def processing_thread():
    global is_running
    # Carregamento do modelo (uma única vez)
    print(f"Carregando {MODEL_SIZE} na GPU...")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    
    with open("reuniao_pt.txt", "a", encoding="utf-8") as f_pt, \
         open("reuniao_en.txt", "a", encoding="utf-8") as f_en:
        
        while is_running or not audio_queue.empty():
            try:
                audio_data = audio_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Processamento com o máximo de contexto (Sem janelas curtas)
            segments_pt, _ = model.transcribe(audio_data, language="pt", vad_filter=True)
            text_pt = " ".join([s.text for s in segments_pt]).strip()

            if text_pt:
                # Tradução (Usando o mesmo áudio para eficiência)
                segments_en, _ = model.transcribe(audio_data, task="translate", vad_filter=True)
                text_en = " ".join([s.text for s in segments_en]).strip()

                print(f"\n[TEXTO]: {text_pt}\n[TRAD]: {text_en}\n")
                f_pt.write(text_pt + "\n")
                f_en.write(text_en + "\n")
                f_pt.flush()
                f_en.flush()

if __name__ == "__main__":
    # Necessário instalar: pip install scipy
    threading.Thread(target=audio_capture_thread, daemon=True).start()
    processing_thread()