#CURL REQUEST:
#curl -X POST -F "video=@/Users/lv/Downloads/test.mp4" http://localhost:5001/analyze-video
#IT RETURNS A JSON OBJECT WITH THE FOLLOWING FIELDS:


import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import openai
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
from google.cloud import speech_v1p1beta1
from datetime import datetime
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import threading
from moviepy.editor import VideoFileClip

# Load environment variables from .env
load_dotenv()

# Configure OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
openai.api_key = OPENAI_API_KEY

# Path to Google credentials JSON
credentials_path = "backend/google_credentials.json"
if not os.path.exists(credentials_path):
    raise ValueError(f"Credentials file not found at: {credentials_path}")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Global list to store confidence scores (if needed later)
confidence_scores = []

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
CORS(app)

# Increase ALL possible limits
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['MAX_FILE_SIZE'] = 100 * 1024 * 1024      # 100MB
app.config['REQUEST_SIZE_LIMIT'] = 100 * 1024 * 1024  # 100MB

# Ensure upload folder exists with proper permissions
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, mode=0o777)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'mkv'}

# At the top with other configs
CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB total

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/analyze-video', methods=['POST'])
def analyze_video_endpoint():
    """Endpoint to handle video upload and analysis."""
    print("=== DEBUG INFO ===")
    print(f"Content Length: {request.content_length}")
    print(f"Content Type: {request.content_type}")
    print(f"Request Headers: {dict(request.headers)}")
    print(f"Request Files: {request.files}")
    print(f"App Config: {app.config}")
    print("================")

    try:
        # Check content length early
        content_length = request.content_length
        if content_length:
            print(f"Content Length: {content_length} bytes")
            print(f"Max File Size: {MAX_FILE_SIZE} bytes")
            if content_length > MAX_FILE_SIZE:
                print(f"Content length {content_length} exceeds max file size {MAX_FILE_SIZE}")
                return jsonify({
                    'success': False,
                    'error': f'File too large. Maximum size is {MAX_FILE_SIZE / (1024 * 1024)}MB'
                }), 413

        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file uploaded'}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400

        filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Ensure the upload folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        print(f"2. Saving file to: {video_path}")
        # Save file in chunks
        with open(video_path, 'wb') as f:
            while True:
                chunk = video_file.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)

        print("3. File saved successfully")

        try:
            print("4. Starting video processing")
            results = process_video(video_path)
            
            if isinstance(results, dict) and 'error' in results:
                return jsonify({
                    'success': False,
                    'error': results['error']
                }), 400

            return jsonify({
                'success': True,
                'results': results
            })

        except MemoryError:
            print("Memory error during processing")
            return jsonify({
                'success': False,
                'error': 'Video file too large to process. Please try a smaller file.'
            }), 413

        except Exception as e:
            print(f"Processing error: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

        finally:
            # Clean up the uploaded file
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
                    print("6. Cleaned up video file")
            except Exception as e:
                print(f"Error cleaning up file: {str(e)}")

    except Exception as e:
        error_msg = str(e)
        print(f"Final error: {error_msg}")
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500


def record_audio():
    # Flag to control recording
    recording_flag = threading.Event()
    recorded_data = []
    sample_rate = 44100

    def callback(indata, frames, time, status):
        if status:
            print(f"Recording status: {status}")
        print(f"Recording callback triggered with {frames} frames")
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

    # Ensure the backend folder exists
    os.makedirs("backend", exist_ok=True)

    print(f"Saving audio to: {audio_path}")
    write(audio_path, sample_rate, recording)
    print(f"Audio file size: {os.path.getsize(audio_path)} bytes")
    return audio_path

def analyze_speech(audio_file_path, cleanup=True):
    client = speech_v1p1beta1.SpeechClient()

    print(f"Analyzing speech for file: {audio_file_path}")
    with open(audio_file_path, 'rb') as audio_file:
        content = audio_file.read()

    print(f"Audio file loaded, size: {len(content)} bytes")

    # Split audio into 10MB chunks
    CHUNK_SIZE = 10 * 1024 * 1024  # 10MB
    chunks = [content[i:i + CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]
    print(f"Split audio into {len(chunks)} chunks")

    all_results = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1} of {len(chunks)}")
        audio = speech_v1p1beta1.RecognitionAudio(content=chunk)
        config = speech_v1p1beta1.RecognitionConfig(
            encoding=speech_v1p1beta1.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="en-US",
            enable_word_confidence=True,
            enable_word_time_offsets=True,
            # Add these options to improve recognition
            use_enhanced=True,
            model='video',  # Specifically for video content
            audio_channel_count=2  # Stereo audio
        )

        try:
            print(f"Sending chunk {i+1} to Google Speech-to-Text API")
            response = client.recognize(config=config, audio=audio)
            print(f"Response received for chunk {i+1}")
            print(f"Results in response: {len(response.results)}")
            
            if response.results:
                all_results.extend(response.results)
                print(f"Chunk {i+1} processed successfully with {len(response.results)} results")
            else:
                print(f"No speech detected in chunk {i+1}")
                
        except Exception as e:
            print(f"Error processing chunk {i+1}: {str(e)}")
            continue

    print(f"Total results collected: {len(all_results)}")

    # Process all results
    transcript = ""
    words = []
    
    for result in all_results:
        if len(result.alternatives) > 0:
            alternative = result.alternatives[0]
            print(f"Found transcript: {alternative.transcript}")
            transcript += alternative.transcript + " "
            words.extend(alternative.words)

    # Calculate metrics
    if words:
        print(f"Total words detected: {len(words)}")
        duration = words[-1].end_time.total_seconds() - words[0].start_time.total_seconds()
        wpm = (len(words) / duration) * 60 if duration > 0 else 0
        avg_confidence = sum(word.confidence for word in words) / len(words)
        
        return {
            'transcript': transcript.strip(),
            'wpm': wpm,
            'confidence': avg_confidence,
            'duration': duration,
            'word_count': len(words)
        }
    else:
        print("No words detected in any chunks")
        return {'error': 'No speech detected in the audio. Please ensure the video contains clear audio.'}

def extract_audio_from_video(video_path):
    """Extract audio from video file and save as WAV."""
    try:
        # Create a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = os.path.join("backend", f"audio_{timestamp}.wav")

        # Ensure 'backend' folder exists
        os.makedirs("backend", exist_ok=True)

        print(f"Extracting audio from: {video_path}")
        video = VideoFileClip(video_path)
        audio = video.audio

        print(f"Video duration: {video.duration:.2f}s, Audio FPS: {audio.fps}")
        
        # Write audio to WAV file
        audio.write_audiofile(audio_path)

        # Close video & audio to free resources
        video.close()
        audio.close()

        print(f"Audio extracted to: {audio_path}")
        return audio_path

    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return None

def process_video(video_path):
    """Process a video file for speech analysis."""
    try:
        print("a. Starting audio extraction")
        audio_path = extract_audio_from_video(video_path)
        if not audio_path:
            print("b. Audio extraction failed")
            return None

        print(f"Audio path is: {audio_path}")
        print("c. Starting speech analysis")
        speech_results = analyze_speech(audio_path)
        print("d. Speech analysis complete")

        if 'error' in speech_results:
            return speech_results

        # Process transcript with ChatGPT
        print("e. Starting ChatGPT analysis")
        transcript = speech_results['transcript']
        
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert speech analyzer. Analyze the following transcript for key points, sentiment, and speaking style."},
                {"role": "user", "content": f"Analyze this transcript: {transcript}"}
            ],
            temperature=0.7,
            max_tokens=500
        )

        # Combine speech-to-text and ChatGPT results
        analysis = response.choices[0].message.content
        
        return {
            **speech_results,  # Include original speech analysis
            'ai_analysis': analysis  # Add ChatGPT analysis
        }

    except Exception as e:
        print(f"Process video error: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)


#curl -X POST -F "video=@/Users/lv/Downloads/test.mp4" http://localhost:5001/analyze-video