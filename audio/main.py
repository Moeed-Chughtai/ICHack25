import subprocess
import json
import torch
from pathlib import Path

# You may need to install these packages:
# pip install ffmpeg-python transformers torch pyannote.audio librosa

from transformers import pipeline
from pyannote.audio import Pipeline as DiarisationPipeline

def convert_video_to_audio(video_path, audio_path="video_audio.wav"):
    """
    Converts a video file to a mono WAV audio file (16kHz).
    """
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ]
    subprocess.run(command, check=True)
    return audio_path

def transcribe_audio_whisper(audio_path, whisper_model="openai/whisper-base"):
    """
    Transcribes audio using a Whisper model from Hugging Face.
    For audio longer than ~30 seconds, chunk_length_s is used 
    to process the audio in smaller segments and merge the text.
    """
    transcriber = pipeline(
        task="automatic-speech-recognition",
        model=whisper_model,
        chunk_length_s=30  # ensures large audio is processed in smaller chunks
    )
    # The pipeline merges the chunks, producing a single text result
    transcription = transcriber(audio_path)["text"]
    return transcription

def perform_speaker_diarisation(audio_path, diarisation_model="pyannote/speaker-diarization"):
    """
    Performs speaker diarisation using a Pyannote pipeline.
    Returns a list of segments with (start_time, end_time, speaker_label).
    """
    diarisation_pipeline = DiarisationPipeline.from_pretrained(diarisation_model)
    diarisation_result = diarisation_pipeline(audio_path)

    segments = []
    for turn, _, speaker_label in diarisation_result.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker_label))

    # Sort segments chronologically
    segments.sort(key=lambda x: x[0])
    return segments

def assign_speakers(segments):
    """
    A simple heuristic that labels the speaker with the most
    talk time as 'Teacher' and all others as 'Student'.
    """
    from collections import defaultdict

    duration_per_speaker = defaultdict(float)
    for start, end, spk in segments:
        duration_per_speaker[spk] += end - start

    teacher_speaker = max(duration_per_speaker, key=duration_per_speaker.get)

    labelled_segments = []
    for (start, end, spk) in segments:
        role = "Teacher" if spk == teacher_speaker else "Student"
        labelled_segments.append((start, end, spk, role))

    return labelled_segments


def main():
    video_path = "video.mp4"
    audio_path = "video_audio.wav"
    output_json_path = "lecture_stats.json"

    # Step 1: Convert video to audio
    audio_path = convert_video_to_audio(video_path, audio_path)

    # Step 2: Transcribe audio with Whisper (with chunking to handle long audio)
    transcript = transcribe_audio_whisper(audio_path)

    # Step 3: Speaker diarisation
    segments = perform_speaker_diarisation(audio_path)
    labelled_segments = assign_speakers(segments)

    # Step 4: Calculate total talk time for Teacher vs Students
    from collections import defaultdict
    time_spoken = defaultdict(float)
    for (start, end, spk_id, role) in labelled_segments:
        time_spoken[role] += (end - start)

    # Step 6: Prepare data for JSON
    data = {
        "transcript": transcript,
        "segments": [
            {
                "start_time": seg[0],
                "end_time": seg[1],
                "speaker_id": seg[2],
                "role": seg[3]
            }
            for seg in labelled_segments
        ],
        "time_spoken": {
            "Teacher": time_spoken["Teacher"],
            "Students": time_spoken["Student"]
        },
    }

    # Step 7: Write results to JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Done. Statistics saved in {output_json_path}")

if __name__ == "__main__":
    main()
