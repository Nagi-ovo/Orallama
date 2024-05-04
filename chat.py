import wave
import time
import torch
import numpy as np
import pyaudio
import whisper
import ollama
import asyncio
import edge_tts
import subprocess
import yaml


INPUT_DEFAULT_DURATION_SECONDS = 5
INPUT_FORMAT = pyaudio.paInt16
INPUT_CHANNELS = 1
INPUT_RATE = 16000
INPUT_CHUNK = 1024
OLLAMA_REST_HEADERS = {'Content-Type': 'application/json'}

class Assistant:
    def __init__(self):
        self.config = self.init_config()

        self.audio = pyaudio.PyAudio()
        self.VOICE = "zh-CN-XiaoyiNeural"

        print(self.config["messages"]["loadingModel"])
        self.model = whisper.load_model(self.config["whisperRecognition"]["modelPath"])
        self.context = []

        time.sleep(0.5)
        print(self.config["messages"]["pressSpace"])

    def init_config(self):
        with open("config.yaml", "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def display_message(self, text):
        print(text)

    def waveform_from_mic(self) -> np.ndarray:
        print("Recording...")
        stream = self.audio.open(format=INPUT_FORMAT,
                                 channels=INPUT_CHANNELS,
                                 rate=INPUT_RATE,
                                 input=True,
                                 frames_per_buffer=INPUT_CHUNK)
        frames = []
        start_time = time.time()
        while time.time() - start_time < INPUT_DEFAULT_DURATION_SECONDS:
            data = stream.read(INPUT_CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        print("Recording stopped.")
        return np.frombuffer(b''.join(frames), np.int16).astype(np.float32) * (1 / 32768.0)

    def speech_to_text(self, waveform):
        transcript = self.model.transcribe(waveform,
                                           language=self.config["whisperRecognition"]["lang"],
                                           fp16=torch.cuda.is_available())
        text = transcript["text"]
        print('\nMe:\n', text.strip())
        return text

    def ask_ollama(self, prompt):
        messages = [{'role': 'user', 'content': prompt}] if not self.context else self.context + [{'role': 'user', 'content': prompt}]
        
        # Assuming ollama.chat is asynchronous and supports streaming
        stream = ollama.chat(
            model=self.config["ollama"]["model"],
            messages=messages,
            stream=True
        )

        full_response = ""
        try:
            # Iterate over each response chunk in the stream
            for chunk in stream:
                # Here, we assume that 'chunk' is a dictionary that contains the message part
                if 'message' in chunk and 'content' in chunk['message']:
                    message_content = chunk['message']['content']
                    full_response += message_content
                    # Optionally, process the message part as it arrives (e.g., for real-time display)

            # After collecting the full response, convert it to speech
            if full_response:
                asyncio.run(self.ollama_response_to_speech(full_response))

            # Update the context for the next request
            self.context.append({'role': 'system', 'content': full_response})
        except Exception as e:
            print(f"Error while streaming response from ollama: {e}")
            # Handle any errors that occur during the streaming process

    async def ollama_response_to_speech(self, response):
        communicate = edge_tts.Communicate(response, self.VOICE)
        temp_file = "./temp_ollama_response.wav"
        final_file = "./final_ollama_response.wav"  # 最终的WAV文件
        try:
            with open(temp_file, "wb") as file:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        file.write(chunk["data"])
            # 使用FFmpeg确保文件是正确的WAV格式
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_file, '-ar', '16000', '-ac', '1', final_file
            ], check=True)
        except Exception as e:
            print(f"Error during audio processing: {e}")
            return

        # 播放生成的音频文件
        try:
            self.play_audio(final_file)
        except Exception as e:
            print(f"Error playing audio file: {e}")

    def play_audio(self, file_path):
        """
        Play audio file.
        """
        wf = wave.open(file_path, 'rb')

        stream = self.audio.open(format=self.audio.get_format_from_width(wf.getsampwidth()),
                                 channels=wf.getnchannels(),
                                 rate=wf.getframerate(),
                                 output=True)

        chunkSize = 1024
        chunk = wf.readframes(chunkSize)
        while chunk:
            stream.write(chunk)
            chunk = wf.readframes(chunkSize)

        wf.close()

def main():
    assistant = Assistant()
    while True:
        input("Press Enter to start recording...")
        speech = assistant.waveform_from_mic()
        transcription = assistant.speech_to_text(waveform=speech)
        assistant.ask_ollama(transcription)
        input("Press Enter to continue...")


if __name__ == "__main__":
    main()
