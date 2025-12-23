"""
Core transcription and diarization functionality.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

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

from .embeddings import extract_embedding_from_segment, match_speaker
from .speakers import get_speaker_name


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def transcribe(
    audio_path: Path,
    model_name: str = "small",
    hf_token: Optional[str] = None
) -> dict:
    """
    Transcribe an audio file with speaker diarization.

    Args:
        audio_path: Path to audio file
        model_name: Whisper model size
        hf_token: HuggingFace token (uses HF_TOKEN env var if not provided)

    Returns:
        Dict with segments, each containing start, end, text, speaker
    """
    # WhisperX uses CTranslate2 which only supports CPU and CUDA
    whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
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
    token = hf_token or os.getenv("HF_TOKEN")
    if not token:
        print("Warning: HF_TOKEN not set. Skipping speaker diarization.")
        return {"segments": result["segments"], "audio": audio}

    print("Running speaker diarization...")
    diarize_model = DiarizationPipeline(
        use_auth_token=token,
        device=pyannote_device
    )
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Include audio data for embedding extraction
    result["audio"] = audio

    return result


def transcribe_with_speakers(
    audio_path: Path,
    profiles: dict,
    model_name: str = "small",
    threshold: float = 0.7,
    hf_token: Optional[str] = None
) -> dict:
    """
    Transcribe audio with automatic speaker recognition.

    Args:
        audio_path: Path to audio file
        profiles: Speaker profiles dict
        model_name: Whisper model size
        threshold: Similarity threshold for speaker matching
        hf_token: HuggingFace token

    Returns:
        Dict with segments including matched speaker names and confidence
    """
    # First do regular transcription
    result = transcribe(audio_path, model_name, hf_token)

    if not profiles.get("profiles"):
        return result

    audio = result.get("audio")
    if audio is None:
        return result

    # Collect segments by speaker for embedding extraction
    speaker_segments = {}
    for segment in result.get("segments", []):
        speaker_id = segment.get("speaker", "UNKNOWN")
        if speaker_id not in speaker_segments:
            speaker_segments[speaker_id] = []
        speaker_segments[speaker_id].append(segment)

    # Extract embeddings and match speakers
    speaker_matches = {}
    print("Matching speakers against profiles...")

    for speaker_id, segments in speaker_segments.items():
        if speaker_id == "UNKNOWN":
            continue

        # Find longest segment for best embedding quality
        best_segment = max(
            segments,
            key=lambda s: s.get("end", 0) - s.get("start", 0)
        )

        start = best_segment.get("start", 0)
        end = best_segment.get("end", 0)

        # Need at least 1 second for reliable embedding
        if end - start < 1.0:
            continue

        embedding = extract_embedding_from_segment(audio, start, end)
        if embedding is not None:
            matched_id, confidence = match_speaker(embedding, profiles, threshold)
            if matched_id:
                speaker_matches[speaker_id] = {
                    "profile_id": matched_id,
                    "confidence": confidence
                }
                print(f"  {speaker_id} -> {profiles['profiles'][matched_id]['name']} ({confidence:.2f})")

    # Update segments with matched speakers
    for segment in result.get("segments", []):
        speaker_id = segment.get("speaker", "UNKNOWN")

        if speaker_id in speaker_matches:
            match = speaker_matches[speaker_id]
            segment["matched_profile"] = match["profile_id"]
            segment["match_confidence"] = match["confidence"]
            segment["speaker_name"] = profiles["profiles"][match["profile_id"]]["name"]
        else:
            segment["speaker_name"] = get_speaker_name(speaker_id, profiles)
            segment["match_confidence"] = 0.0

    return result


def export_transcript(
    result: dict,
    audio_path: Path,
    profiles: dict,
    format: Literal["markdown", "json"] = "markdown"
) -> str:
    """
    Export transcription result to markdown or JSON format.

    Args:
        result: Transcription result dict
        audio_path: Original audio file path
        profiles: Speaker profiles dict
        format: Output format ("markdown" or "json")

    Returns:
        Formatted string (markdown or JSON)
    """
    if format == "json":
        return _export_json(result, audio_path, profiles)
    else:
        return _export_markdown(result, audio_path, profiles)


def _export_json(result: dict, audio_path: Path, profiles: dict) -> str:
    """Export to JSON format for GUI consumption."""
    segments = []

    for segment in result.get("segments", []):
        speaker_id = segment.get("speaker", "UNKNOWN")

        segments.append({
            "start": segment.get("start", 0),
            "end": segment.get("end", 0),
            "speaker_id": speaker_id,
            "speaker_name": segment.get("speaker_name", get_speaker_name(speaker_id, profiles)),
            "matched_profile": segment.get("matched_profile"),
            "confidence": segment.get("match_confidence", 0.0),
            "text": segment.get("text", "").strip()
        })

    # Calculate duration from last segment
    duration = max((s["end"] for s in segments), default=0)

    output = {
        "source": audio_path.name,
        "date": datetime.now().isoformat(),
        "duration": duration,
        "segments": segments
    }

    return json.dumps(output, indent=2)


def _export_markdown(result: dict, audio_path: Path, profiles: dict) -> str:
    """Export to markdown format."""
    lines = []

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
        speaker_id = segment.get("speaker", "UNKNOWN")
        speaker_name = segment.get("speaker_name", get_speaker_name(speaker_id, profiles))
        text = segment.get("text", "").strip()
        start = segment.get("start", 0)

        if not text:
            continue

        # Group consecutive segments from same speaker
        if speaker_id == current_speaker:
            current_text.append(text)
        else:
            # Output previous speaker's text
            if current_speaker is not None and current_text:
                prev_name = current_text_speaker_name
                timestamp = _format_timestamp(current_start)
                combined_text = " ".join(current_text)
                lines.append(f"**{timestamp} {prev_name}:**")
                lines.append(combined_text)
                lines.append("")

            # Start new speaker
            current_speaker = speaker_id
            current_text = [text]
            current_text_speaker_name = speaker_name
            current_start = start

    # Output final speaker's text
    if current_speaker is not None and current_text:
        timestamp = _format_timestamp(current_start)
        combined_text = " ".join(current_text)
        lines.append(f"**{timestamp} {current_text_speaker_name}:**")
        lines.append(combined_text)
        lines.append("")

    return "\n".join(lines)


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to [HH:MM:SS] or [MM:SS] format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"
    return f"[{minutes:02d}:{secs:02d}]"
