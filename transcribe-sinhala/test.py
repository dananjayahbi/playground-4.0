import os
from transformers import pipeline
import io

# Path to your local audio file (use .mp3, .wav, .m4a, etc.)
audio_path = "t1.wav"

# Create output text file path with same name as audio file
output_text_file = os.path.splitext(audio_path)[0] + ".txt"

# Initialize the pipeline with the specified model
pipe = pipeline(model="Lingalingeswaran/whisper-small-sinhala")

# Transcribe the audio file to text
result = pipe(audio_path)
transcription = result["text"]
print(f"Detected language: si (Sinhala)")

# Print the transcription
print("Transcription:")
print(transcription)

# Save the transcription to a text file with UTF-8 encoding for proper Sinhala font support
with io.open(output_text_file, 'w', encoding='utf-8') as f:
    f.write(transcription)

print(f"Transcription saved to {output_text_file} with UTF-8 encoding for Sinhala font support")
