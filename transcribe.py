#!/usr/bin/env python3
"""
Meeting Transcription Tool - CLI Interface

Transcribes audio files with speaker diarization and automatic speaker recognition.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from transcriber import (
    transcribe_with_speakers,
    export_transcript,
    load_profiles,
    save_profiles,
    enroll_speaker,
    list_speakers,
)

# Load environment variables
load_dotenv()

# Constants
SCRIPT_DIR = Path(__file__).parent
TRANSCRIPTS_DIR = SCRIPT_DIR / "transcripts"
SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}


def cmd_transcribe(args):
    """Handle transcription command."""
    audio_path = args.audio_file

    # Validate input file
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    if audio_path.suffix.lower() not in SUPPORTED_FORMATS:
        print(f"Error: Unsupported format. Supported: {', '.join(SUPPORTED_FORMATS)}")
        sys.exit(1)

    # Ensure output directory exists
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)

    # Determine output path
    if args.output:
        output_path = TRANSCRIPTS_DIR / args.output
    else:
        ext = ".json" if args.format == "json" else ".md"
        output_path = TRANSCRIPTS_DIR / f"{audio_path.stem}{ext}"

    # Load speaker profiles
    profiles = load_profiles()

    # Transcribe with speaker recognition
    print(f"\nTranscribing: {audio_path}")
    print("-" * 40)

    result = transcribe_with_speakers(
        audio_path,
        profiles,
        model_name=args.model,
        threshold=args.threshold
    )

    # Export transcript
    output = export_transcript(result, audio_path, profiles, format=args.format)

    # Save transcript
    with open(output_path, "w") as f:
        f.write(output)

    print("-" * 40)
    print(f"Transcript saved: {output_path}")

    # Report new speakers
    seen_speakers = set()
    for segment in result.get("segments", []):
        speaker_id = segment.get("speaker", "UNKNOWN")
        if speaker_id != "UNKNOWN" and not segment.get("matched_profile"):
            seen_speakers.add(speaker_id)

    if seen_speakers:
        print(f"\nUnrecognized speakers: {', '.join(sorted(seen_speakers))}")
        print("Use --enroll to add voice profiles for better recognition.")


def cmd_enroll(args):
    """Handle speaker enrollment command."""
    name = args.enroll
    audio_path = args.audio_file

    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    print(f"Enrolling speaker '{name}' from {audio_path}...")

    profiles = enroll_speaker(name, audio_path)

    speaker_id = name.lower().replace(" ", "_")
    profile = profiles["profiles"][speaker_id]
    embedding_count = len(profile["embeddings"])

    print(f"Success! '{name}' now has {embedding_count} voice sample(s).")
    print("Tip: Enroll multiple samples for better recognition accuracy.")


def cmd_list_speakers(args):
    """Handle list speakers command."""
    speakers = list_speakers()

    if not speakers:
        print("No speakers enrolled yet.")
        print("Use --enroll to add voice profiles.")
        return

    print(f"\nEnrolled Speakers ({len(speakers)}):")
    print("-" * 40)

    for speaker in speakers:
        samples = speaker["embedding_count"]
        print(f"  {speaker['name']}")
        print(f"    ID: {speaker['id']}")
        print(f"    Samples: {samples}")
        print(f"    Created: {speaker['created'][:10]}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files with speaker diarization and recognition"
    )

    # Positional argument (optional for some commands)
    parser.add_argument(
        "audio_file",
        type=Path,
        nargs="?",
        help="Path to audio file (mp3, wav, m4a, flac, ogg, webm)"
    )

    # Transcription options
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
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Speaker matching threshold 0-1 (default: 0.7)"
    )

    # Speaker management
    parser.add_argument(
        "--enroll",
        metavar="NAME",
        help="Enroll speaker with given name from audio file"
    )
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="List all enrolled speakers"
    )

    args = parser.parse_args()

    # Route to appropriate command
    if args.list_speakers:
        cmd_list_speakers(args)
    elif args.enroll:
        if not args.audio_file:
            print("Error: Audio file required for enrollment")
            print("Usage: python transcribe.py --enroll 'Name' audio/sample.mp3")
            sys.exit(1)
        cmd_enroll(args)
    else:
        if not args.audio_file:
            parser.print_help()
            sys.exit(1)
        cmd_transcribe(args)


if __name__ == "__main__":
    main()
