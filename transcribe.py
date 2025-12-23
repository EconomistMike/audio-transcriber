#!/usr/bin/env python3
"""
Meeting Transcription Tool
Transcribes audio files with speaker diarization using WhisperX.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

# PyTorch 2.6+ defaults to weights_only=True which breaks pyannote model loading.
# Monkey-patch to force weights_only=False for trusted HuggingFace models.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import whisperx
from whisperx.diarize import DiarizationPipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
SCRIPT_DIR = Path(__file__).parent
SPEAKER_MAP_FILE = SCRIPT_DIR / "speaker_map.json"
TRANSCRIPTS_DIR = SCRIPT_DIR / "transcripts"
SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}


def get_device():
    """Determine the best available device for inference."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_speaker_map() -> dict:
    """Load speaker name mappings from JSON file."""
    if SPEAKER_MAP_FILE.exists():
        with open(SPEAKER_MAP_FILE) as f:
            return json.load(f)
    return {"_embeddings": {}}


def save_speaker_map(speaker_map: dict):
    """Save speaker name mappings to JSON file."""
    with open(SPEAKER_MAP_FILE, "w") as f:
        json.dump(speaker_map, f, indent=2)


def get_speaker_name(speaker_id: str, speaker_map: dict) -> str:
    """Get human-readable name for a speaker ID."""
    return speaker_map.get(speaker_id, speaker_id)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to [HH:MM:SS] format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"
    return f"[{minutes:02d}:{secs:02d}]"


def transcribe_audio(audio_path: Path, model_name: str = "small") -> dict:
    """
    Transcribe audio file with speaker diarization.

    Returns dict with segments containing speaker labels and timestamps.
    """
    # WhisperX uses CTranslate2 which only supports CPU and CUDA
    whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
    # Pyannote can use MPS on Apple Silicon
    pyannote_device = get_device()
    compute_type = "float16" if whisper_device == "cuda" else "float32"

    print(f"Using device: {whisper_device} (whisper), {pyannote_device} (diarization)")
    print(f"Loading WhisperX model ({model_name})...")

    # Load model
    model = whisperx.load_model(model_name, whisper_device, compute_type=compute_type)

    # Transcribe
    print("Transcribing audio...")
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=16)

    # Align whisper output
    print("Aligning transcription...")
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=whisper_device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        whisper_device,
        return_char_alignments=False
    )

    # Speaker diarization
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not set. Skipping speaker diarization.")
        print("Set HF_TOKEN environment variable or add it to .env file.")
        return result

    print("Running speaker diarization...")
    diarize_model = DiarizationPipeline(
        use_auth_token=hf_token,
        device=pyannote_device
    )
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    return result


def generate_markdown(
    result: dict,
    audio_path: Path,
    speaker_map: dict
) -> tuple[str, set]:
    """
    Generate markdown transcript from transcription result.

    Returns (markdown_content, set of new speaker IDs found).
    """
    lines = []
    new_speakers = set()

    # Header
    lines.append("# Meeting Transcript")
    lines.append(f"**Source:** {audio_path.name}")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Process segments, grouping by speaker
    current_speaker = None
    current_text = []
    current_start = None

    for segment in result.get("segments", []):
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "").strip()
        start = segment.get("start", 0)

        if not text:
            continue

        # Track new speakers
        if speaker not in speaker_map and speaker != "UNKNOWN":
            new_speakers.add(speaker)

        # Group consecutive segments from same speaker
        if speaker == current_speaker:
            current_text.append(text)
        else:
            # Output previous speaker's text
            if current_speaker is not None and current_text:
                speaker_name = get_speaker_name(current_speaker, speaker_map)
                timestamp = format_timestamp(current_start)
                combined_text = " ".join(current_text)
                lines.append(f"**{timestamp} {speaker_name}:**")
                lines.append(combined_text)
                lines.append("")

            # Start new speaker
            current_speaker = speaker
            current_text = [text]
            current_start = start

    # Output final speaker's text
    if current_speaker is not None and current_text:
        speaker_name = get_speaker_name(current_speaker, speaker_map)
        timestamp = format_timestamp(current_start)
        combined_text = " ".join(current_text)
        lines.append(f"**{timestamp} {speaker_name}:**")
        lines.append(combined_text)
        lines.append("")

    return "\n".join(lines), new_speakers


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files with speaker diarization"
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to audio file (mp3, wav, m4a, flac, ogg, webm)"
    )
    parser.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: small)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output filename (default: <audio_name>.md)"
    )

    args = parser.parse_args()

    # Validate input file
    if not args.audio_file.exists():
        print(f"Error: File not found: {args.audio_file}")
        sys.exit(1)

    if args.audio_file.suffix.lower() not in SUPPORTED_FORMATS:
        print(f"Error: Unsupported format. Supported: {', '.join(SUPPORTED_FORMATS)}")
        sys.exit(1)

    # Ensure output directory exists
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)

    # Determine output path
    if args.output:
        output_path = TRANSCRIPTS_DIR / args.output
    else:
        output_path = TRANSCRIPTS_DIR / f"{args.audio_file.stem}.md"

    # Load speaker map
    speaker_map = load_speaker_map()

    # Transcribe
    print(f"\nTranscribing: {args.audio_file}")
    print("-" * 40)

    result = transcribe_audio(args.audio_file, args.model)

    # Generate markdown
    markdown, new_speakers = generate_markdown(result, args.audio_file, speaker_map)

    # Save transcript
    with open(output_path, "w") as f:
        f.write(markdown)

    print("-" * 40)
    print(f"Transcript saved: {output_path}")

    # Update speaker map with new speakers
    if new_speakers:
        for speaker in new_speakers:
            if speaker not in speaker_map:
                speaker_map[speaker] = speaker  # Default to ID
        save_speaker_map(speaker_map)
        print(f"\nNew speakers found: {', '.join(sorted(new_speakers))}")
        print(f"Edit {SPEAKER_MAP_FILE} to assign names.")


if __name__ == "__main__":
    main()
