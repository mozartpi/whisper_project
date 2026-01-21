import os
import sys
import queue
import threading
import numpy as np
import pyaudio
from faster_whisper import WhisperModel

# --- CONFIGURAÇÕES AVANÇADAS ---
MODEL_SIZE = "large-v3" # Recomendado para manter latência baixa na 3050
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"
RATE = 16000
CHUNK_DURATION = 3  # Segundos de áudio por bloco de processamento
CHANNELS = 1

audio_queue = queue.Queue()
is_running = True

def audio_capture_thread():
    """Thread dedicada apenas a capturar o som sem interrupções."""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE, 
                    input=True, frames_per_buffer=1024)
    
    print("[INFO] Captura iniciada. Fale agora...")
    
    frames_per_chunk = int(RATE / 1024 * CHUNK_DURATION)
    
    while is_running:
        chunk_data = []
        for _ in range(frames_per_chunk):
            if not is_running: break
            chunk_data.append(stream.read(1024, exception_on_overflow=False))
        
        audio_np = np.frombuffer(b''.join(chunk_data), dtype=np.int16).astype(np.float32) / 32768.0
        audio_queue.put(audio_np)

    stream.stop_stream()
    stream.close()
    p.terminate()

def processing_thread():
    """Thread dedicada a processar a fila com VAD e salvar arquivos."""
    global is_running
    
    # Ativando VAD para eliminar alucinações e silêncio
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    
    with open("transcricao_pt.txt", "a", encoding="utf-8") as f_pt, \
         open("transcricao_en.txt", "a", encoding="utf-8") as f_en:
        
        while is_running or not audio_queue.empty():
            try:
                audio_chunk = audio_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Transcrição com VAD Filter ativado
            segments_pt, _ = model.transcribe(
                audio_chunk, 
                language="pt", 
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Tradução com VAD Filter ativado
            segments_en, _ = model.transcribe(
                audio_chunk, 
                task="translate", 
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            text_pt = "".join([s.text for s in segments_pt]).strip()
            text_en = "".join([s.text for s in segments_en]).strip()

            if text_pt:
                output_pt = f"PT: {text_pt}"
                output_en = f"EN: {text_en}"
                print(f"\n{output_pt}\n{output_en}")
                
                f_pt.write(text_pt + " ")
                f_en.write(text_en + " ")
                # Flush garante que o texto vá para o disco imediatamente
                f_pt.flush()
                f_en.flush()
            
            audio_queue.task_done()

if __name__ == "__main__":
    t1 = threading.Thread(target=audio_capture_thread)
    t2 = threading.Thread(target=processing_thread)

    try:
        t1.start()
        t2.start()
        while True:
            t1.join(0.1)
            t2.join(0.1)
    except KeyboardInterrupt:
        print("\n[ENCERRANDO] Aguardando processamento do buffer final...")
        is_running = False
        t1.join()
        t2.join()
        print("[OK] Tudo salvo com sucesso.")