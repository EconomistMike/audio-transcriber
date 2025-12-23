"""
Audio Transcriber - Core Library

A modular library for audio transcription with speaker diarization
and voice embedding-based speaker recognition.
"""

from .core import transcribe, transcribe_with_speakers, export_transcript
from .speakers import (
    load_profiles,
    save_profiles,
    enroll_speaker,
    get_speaker_name,
    list_speakers,
)
from .embeddings import (
    extract_embedding,
    compare_embeddings,
    match_speaker,
)

__all__ = [
    # Core
    "transcribe",
    "transcribe_with_speakers",
    "export_transcript",
    # Speakers
    "load_profiles",
    "save_profiles",
    "enroll_speaker",
    "get_speaker_name",
    "list_speakers",
    # Embeddings
    "extract_embedding",
    "compare_embeddings",
    "match_speaker",
]
