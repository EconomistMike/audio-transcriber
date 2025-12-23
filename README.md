# Audio Transcriber

A CLI tool for transcribing meeting audio files with speaker diarization using WhisperX.

## Features

- Transcribes audio files (mp3, wav, m4a, flac, ogg, webm)
- Speaker diarization to identify different speakers
- Timestamped markdown output
- Configurable speaker name mapping
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

```bash
# Basic usage
python transcribe.py audio/meeting.mp3

# Specify model size
python transcribe.py audio/meeting.mp3 --model medium

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

## Speaker Mapping

After transcription, edit `speaker_map.json` to assign real names:

```json
{
  "SPEAKER_00": "Mike",
  "SPEAKER_01": "Sarah",
  "_embeddings": {}
}
```

The `_embeddings` key is reserved for future voice recognition features.

## Output Format

Transcripts are saved to `transcripts/` as markdown:

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

## Project Structure

```
audio-transcriber/
├── transcribe.py      # Main CLI script
├── requirements.txt   # Dependencies
├── speaker_map.json   # Speaker name mappings
├── .env               # HuggingFace token (not committed)
├── audio/             # Input audio files
└── transcripts/       # Output markdown files
```

## License

MIT
