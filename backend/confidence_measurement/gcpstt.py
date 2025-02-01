from google.cloud import speech_v1
import os
from google.cloud import speech_v1p1beta1
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import numpy as np
from dotenv import load_dotenv
from google.api_core import client_options
from datetime import datetime
import keyboard
import threading
import openai

load_dotenv()  # Load environment variables from .env file

# Configure OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
openai.api_key = OPENAI_API_KEY

# Use credentials file directly
credentials_path = "./google_credentials.json"
if not os.path.exists(credentials_path):
    raise ValueError("Credentials file not found at: " + credentials_path)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Global list to store confidence scores
confidence_scores = []

def record_audio():
    # Flag to control recording
    recording_flag = threading.Event()
    recorded_data = []
    sample_rate = 44100
    
    def callback(indata, frames, time, status):
        if recording_flag.is_set():
            recorded_data.append(indata.copy())
    
    print("Press Enter to start recording...")
    input()  # Wait for Enter key
    
    recording_flag.set()
    print("Recording... Press Enter again to stop")
    
    # Start recording stream
    with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
        input()  # Wait for Enter key
        recording_flag.clear()
    
    print("Recording finished")
    
    if not recorded_data:
        print("No data recorded!")
        return None
    
    # Combine all recorded chunks
    recording = np.concatenate(recorded_data, axis=0)
    
    # Normalize and convert to int16 for proper WAV format
    recording = np.int16(recording * 32767)
    
    # Create a filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_path = os.path.join("backend", f"recording_{timestamp}.wav")
    
    # Save the WAV file
    print(f"Saving audio to: {audio_path}")
    write(audio_path, sample_rate, recording)
    print(f"Audio file size: {os.path.getsize(audio_path)} bytes")
    return audio_path

def analyze_speech(audio_file_path, cleanup=True):
    client = speech_v1p1beta1.SpeechClient()

    with open(audio_file_path, 'rb') as audio_file:
        content = audio_file.read()

    audio = speech_v1p1beta1.RecognitionAudio(content=content)
    config = speech_v1p1beta1.RecognitionConfig(
        encoding=speech_v1p1beta1.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="en-US",
        enable_word_confidence=True,
        enable_word_time_offsets=True
    )

    response = client.recognize(config=config, audio=audio)
    confidence_score = None

    for result in response.results:
        transcript = result.alternatives[0].transcript
        print("\nTranscript:", transcript)
        
        # Speech analysis
        words = result.alternatives[0].words
        
        # Speech rate calculation
        wpm = 0
        if len(words) > 1:
            duration = words[-1].end_time.total_seconds() - words[0].start_time.total_seconds()
            wpm = (len(words) / duration) * 60
            print(f"Speaking rate: {wpm:.2f} words per minute")

        # Confidence and hesitation analysis
        hesitations = 0
        avg_confidence = 0
        low_confidence_words = []
        word_timings = []
        
        for word in words:
            if word.confidence < 0.8:
                hesitations += 1
                low_confidence_words.append(f"{word.word} ({word.confidence:.2%})")
            avg_confidence += word.confidence
            
            word_duration = word.end_time.total_seconds() - word.start_time.total_seconds()
            word_timings.append(f"{word.word}: {word_duration:.2f}s")

        avg_confidence = avg_confidence / len(words) if words else 0
        
        # Calculate pauses between words
        pauses = []
        for i in range(len(words) - 1):
            pause_duration = words[i + 1].start_time.total_seconds() - words[i].end_time.total_seconds()
            if pause_duration > 0.5:
                pauses.append(f"After '{words[i].word}': {pause_duration:.2f}s")

        print(f"Hesitations detected: {hesitations}")
        print(f"Average confidence: {avg_confidence:.2%}")

        # Create analysis prompt
        chatgpt_prompt = f"""Based on these speech parameters, output only a number from 0 to 100. This is an EXTREMELY STRICT scoring system.

Start with 100 points and apply ALL of these harsh penalties:

MAJOR PENALTIES:
- Average accuracy below 0.95: -40 points
- Average accuracy below 0.90: -60 points
- Average accuracy below 0.85: -80 points
- Speaking rate outside 130-150 wpm: -5 points per 5 wpm deviation
- Each hesitation: -15 points
- Each pause >0.5s: -12 points
- Each low confidence word (<0.8): -8 points

ADDITIONAL PENALTIES:
- More than 2 hesitations in any 30-second period: -20 points
- Any single word with confidence <0.7: -10 points per word
- Any pause >1.0s: additional -15 points per pause
- Irregular speaking pace (high variance in word timing): -25 points

Final score cannot exceed these STRICT maximums based on performance:
- Maximum 95: Perfect delivery (zero hesitations, 135-145 wpm, 98%+ accuracy)
- Maximum 80: 1 hesitation or any accuracy below 95%
- Maximum 60: 2-3 hesitations or any accuracy below 90%
- Maximum 40: 4+ hesitations or any accuracy below 85%
- Maximum 20: 6+ hesitations or any accuracy below 80%

Speech Metrics:
- Average accuracy score: {avg_confidence:.2%}
- Speaking rate: {wpm:.2f} words per minute
- Hesitations: {hesitations}
- Words: {len(words)}
- Low confidence words: {len(low_confidence_words)}
- Significant pauses: {len(pauses)}

Transcript: "{transcript}"

IMPORTANT: Be extremely strict! Most speeches should score below 50. Output format: Just the number (0-100)"""

        print("\nRequesting confidence score...")

        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a speech analyst. Respond only with a number from 0 to 100 representing speaker confidence."},
                    {"role": "user", "content": chatgpt_prompt}
                ]
            )
            confidence_score = int(response.choices[0].message.content.strip())
            confidence_scores.append(confidence_score)
            print(f"\nConfidence Score: {confidence_score}/100")
            
            if len(confidence_scores) > 0:
                avg_score = sum(confidence_scores) / len(confidence_scores)
                print(f"Average Confidence Score: {avg_score:.2f}/100")
        except Exception as e:
            print(f"Error getting confidence score: {str(e)}")

    if cleanup:
        os.unlink(audio_file_path)

    return {
        "transcript": transcript if 'transcript' in locals() else None,
        "confidence_score": confidence_score,
        "average_confidence_score": sum(confidence_scores) / len(confidence_scores) if confidence_scores else None,
        "speech_rate": wpm if 'wpm' in locals() else None,
        "hesitations": hesitations if 'hesitations' in locals() else None,
        "duration": duration if 'duration' in locals() else None,
        "avg_confidence": avg_confidence if 'avg_confidence' in locals() else None
    }

if __name__ == "__main__":
    audio_file = record_audio()
    analyze_speech(audio_file)
