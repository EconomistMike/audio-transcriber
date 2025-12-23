"""
Speaker profile management.

Handles loading, saving, and managing speaker profiles with voice embeddings.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .embeddings import extract_embedding, embedding_to_base64

# Default paths
_DEFAULT_PROFILES_PATH = Path(__file__).parent.parent / "speaker_profiles.json"


def _get_profiles_path() -> Path:
    """Get the speaker profiles file path."""
    return _DEFAULT_PROFILES_PATH


def load_profiles(path: Optional[Path] = None) -> dict:
    """
    Load speaker profiles from JSON file.

    Args:
        path: Optional custom path to profiles file

    Returns:
        Dict containing profiles and aliases
    """
    profiles_path = path or _get_profiles_path()

    if profiles_path.exists():
        with open(profiles_path) as f:
            return json.load(f)

    return {
        "profiles": {},
        "aliases": {}
    }


def save_profiles(profiles: dict, path: Optional[Path] = None):
    """
    Save speaker profiles to JSON file.

    Args:
        profiles: Profiles dict to save
        path: Optional custom path to profiles file
    """
    profiles_path = path or _get_profiles_path()

    with open(profiles_path, "w") as f:
        json.dump(profiles, f, indent=2)


def enroll_speaker(
    name: str,
    audio_path: Path,
    profiles: Optional[dict] = None,
    save: bool = True
) -> dict:
    """
    Enroll a new speaker or add embedding to existing profile.

    Args:
        name: Display name for the speaker
        audio_path: Path to audio sample of the speaker
        profiles: Existing profiles dict (loads from file if None)
        save: Whether to save profiles after enrollment

    Returns:
        Updated profiles dict
    """
    if profiles is None:
        profiles = load_profiles()

    # Generate speaker ID from name
    speaker_id = name.lower().replace(" ", "_")

    # Extract embedding
    embedding = extract_embedding(audio_path)
    embedding_b64 = embedding_to_base64(embedding)

    # Create or update profile
    if speaker_id not in profiles["profiles"]:
        profiles["profiles"][speaker_id] = {
            "name": name,
            "embeddings": [],
            "created": datetime.now().isoformat(),
            "sample_files": []
        }

    # Add embedding and sample file
    profiles["profiles"][speaker_id]["embeddings"].append(embedding_b64)
    profiles["profiles"][speaker_id]["sample_files"].append(str(audio_path.name))

    if save:
        save_profiles(profiles)

    return profiles


def remove_speaker(speaker_id: str, profiles: Optional[dict] = None, save: bool = True) -> dict:
    """
    Remove a speaker profile.

    Args:
        speaker_id: ID of speaker to remove
        profiles: Existing profiles dict (loads from file if None)
        save: Whether to save profiles after removal

    Returns:
        Updated profiles dict
    """
    if profiles is None:
        profiles = load_profiles()

    # Remove profile
    if speaker_id in profiles["profiles"]:
        del profiles["profiles"][speaker_id]

    # Remove any aliases pointing to this speaker
    profiles["aliases"] = {
        k: v for k, v in profiles["aliases"].items()
        if v != speaker_id
    }

    if save:
        save_profiles(profiles)

    return profiles


def get_speaker_name(speaker_id: str, profiles: dict) -> str:
    """
    Get display name for a speaker ID.

    Checks aliases first, then profiles, then returns the raw ID.

    Args:
        speaker_id: Speaker ID (e.g., "SPEAKER_00" or profile ID)
        profiles: Profiles dict

    Returns:
        Display name for the speaker
    """
    # Check if it's an alias
    if speaker_id in profiles.get("aliases", {}):
        profile_id = profiles["aliases"][speaker_id]
        if profile_id in profiles.get("profiles", {}):
            return profiles["profiles"][profile_id]["name"]

    # Check if it's a direct profile ID
    if speaker_id in profiles.get("profiles", {}):
        return profiles["profiles"][speaker_id]["name"]

    return speaker_id


def set_alias(
    speaker_id: str,
    profile_id: str,
    profiles: Optional[dict] = None,
    save: bool = True
) -> dict:
    """
    Set an alias mapping a temporary speaker ID to a profile.

    Args:
        speaker_id: Temporary speaker ID (e.g., "SPEAKER_00")
        profile_id: Profile ID to map to
        profiles: Existing profiles dict (loads from file if None)
        save: Whether to save profiles after setting alias

    Returns:
        Updated profiles dict
    """
    if profiles is None:
        profiles = load_profiles()

    profiles["aliases"][speaker_id] = profile_id

    if save:
        save_profiles(profiles)

    return profiles


def list_speakers(profiles: Optional[dict] = None) -> list[dict]:
    """
    List all enrolled speakers.

    Args:
        profiles: Existing profiles dict (loads from file if None)

    Returns:
        List of speaker info dicts with id, name, embedding_count, created
    """
    if profiles is None:
        profiles = load_profiles()

    speakers = []
    for speaker_id, profile in profiles.get("profiles", {}).items():
        speakers.append({
            "id": speaker_id,
            "name": profile["name"],
            "embedding_count": len(profile.get("embeddings", [])),
            "sample_files": profile.get("sample_files", []),
            "created": profile.get("created", "unknown")
        })

    return sorted(speakers, key=lambda x: x["name"])


def migrate_from_speaker_map(speaker_map_path: Path) -> dict:
    """
    Migrate from old speaker_map.json format to new profiles format.

    Args:
        speaker_map_path: Path to old speaker_map.json

    Returns:
        New profiles dict
    """
    profiles = load_profiles()

    if not speaker_map_path.exists():
        return profiles

    with open(speaker_map_path) as f:
        old_map = json.load(f)

    # Migrate manual mappings as aliases
    for speaker_id, name in old_map.items():
        if speaker_id.startswith("_"):
            continue  # Skip special keys like _embeddings

        if speaker_id != name:  # Only if actually mapped to a different name
            # Create a profile for this name if it doesn't exist
            profile_id = name.lower().replace(" ", "_")
            if profile_id not in profiles["profiles"]:
                profiles["profiles"][profile_id] = {
                    "name": name,
                    "embeddings": [],
                    "created": datetime.now().isoformat(),
                    "sample_files": []
                }
            # Set alias
            profiles["aliases"][speaker_id] = profile_id

    return profiles
