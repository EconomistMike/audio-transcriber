# Audio Transcriber

A CLI tool for transcribing meeting audio files with speaker diarization and automatic speaker recognition using voice embeddings.

## Features

- Transcribes audio files (mp3, wav, m4a, flac, ogg, webm)
- Speaker diarization to identify different speakers
- **Voice embeddings for automatic speaker recognition**
- Timestamped markdown or JSON output
- Apple Silicon (M1/M2/M3) compatible

## Installation

```bash
git clone https://github.com/EconomistMike/audio-transcriber.git
cd audio-transcriber
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Setup

### 1. HuggingFace Token

Get a token from https://huggingface.co/settings/tokens and add it to `.env`:

```bash
echo "HF_TOKEN=your_token_here" > .env
```

### 2. Accept Pyannote Model Terms

Visit these links and accept the terms (required for speaker diarization):

- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

## Usage

### Basic Transcription

```bash
python transcribe.py audio/meeting.mp3
```

### Speaker Enrollment

Enroll speakers for automatic recognition:

```bash
# Enroll a speaker from an audio sample
python transcribe.py --enroll "Mike" audio/mike-sample.mp3

# Add more samples for better accuracy
python transcribe.py --enroll "Mike" audio/mike-sample2.mp3

# List enrolled speakers
python transcribe.py --list-speakers
```

### Advanced Options

```bash
# Specify model size
python transcribe.py audio/meeting.mp3 --model medium

# Export as JSON (for GUI/API consumption)
python transcribe.py audio/meeting.mp3 --format json

# Adjust speaker matching threshold (0-1, default: 0.7)
python transcribe.py audio/meeting.mp3 --threshold 0.6

# Custom output filename
python transcribe.py audio/meeting.mp3 --output my-meeting.md
```

### Model Options

| Model | Speed | Accuracy | Memory |
|-------|-------|----------|--------|
| tiny | Fastest | Lower | ~1GB |
| base | Fast | Good | ~1GB |
| small | Medium | Better | ~2GB |
| medium | Slow | High | ~5GB |
| large-v3 | Slowest | Highest | ~10GB |

## Output Formats

### Markdown (default)

```markdown
# Meeting Transcript
**Source:** meeting.mp3
**Date:** 2025-12-23

---

**[00:00:05] Mike:**
Hello everyone, thanks for joining today's meeting.

**[00:00:12] Sarah:**
Thanks for having us. Let's get started.
```

### JSON (for GUI/API)

```json
{
  "source": "meeting.mp3",
  "date": "2025-12-23T10:00:00",
  "duration": 3600.5,
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "speaker_id": "SPEAKER_00",
      "speaker_name": "Mike",
      "confidence": 0.85,
      "text": "Hello everyone..."
    }
  ]
}
```

## Project Structure

```
audio-transcriber/
├── transcribe.py           # CLI interface
├── transcriber/            # Core library
│   ├── core.py            # Transcription & export
│   ├── embeddings.py      # Voice embedding extraction
│   └── speakers.py        # Speaker profile management
├── speaker_profiles.json   # Enrolled speaker data
├── requirements.txt        # Dependencies
├── audio/                  # Input audio files
└── transcripts/            # Output files
```

## Library API

The `transcriber` module can be imported directly for GUI or API integration:

```python
from transcriber import (
    transcribe_with_speakers,
    export_transcript,
    enroll_speaker,
    load_profiles,
    list_speakers,
)

# Load profiles
profiles = load_profiles()

# Transcribe with speaker recognition
result = transcribe_with_speakers("meeting.mp3", profiles, threshold=0.7)

# Export to JSON
json_output = export_transcript(result, "meeting.mp3", profiles, format="json")
```

## License

MIT
