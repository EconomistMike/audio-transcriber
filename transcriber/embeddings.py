"""
Voice embedding extraction and speaker matching.

Uses SpeechBrain's ECAPA-TDNN model for speaker embeddings.
"""

import base64
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

# Lazy-loaded model
_embedding_model = None
_embedding_device = None


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_embedding_model():
    """Lazy-load the SpeechBrain embedding model."""
    global _embedding_model, _embedding_device

    if _embedding_model is None:
        from speechbrain.inference.speaker import EncoderClassifier

        _embedding_device = get_device()
        # Use CPU for SpeechBrain as MPS can have issues
        run_device = "cpu" if _embedding_device == "mps" else _embedding_device

        _embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/spkrec-ecapa-voxceleb",
            run_opts={"device": run_device}
        )

    return _embedding_model


def extract_embedding(audio_path: Path) -> np.ndarray:
    """
    Extract speaker embedding from an audio file.

    Args:
        audio_path: Path to audio file

    Returns:
        numpy array of shape (192,) containing the speaker embedding
    """
    model = _load_embedding_model()

    # Load and resample audio to 16kHz (required by model)
    waveform, sample_rate = torchaudio.load(str(audio_path))

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # Convert stereo to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Extract embedding
    with torch.no_grad():
        embedding = model.encode_batch(waveform)

    return embedding.squeeze().cpu().numpy()


def extract_embedding_from_segment(
    audio: np.ndarray,
    start: float,
    end: float,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Extract speaker embedding from an audio segment.

    Args:
        audio: Audio waveform as numpy array
        start: Start time in seconds
        end: End time in seconds
        sample_rate: Sample rate of the audio

    Returns:
        numpy array of shape (192,) containing the speaker embedding
    """
    model = _load_embedding_model()

    # Extract segment
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    segment = audio[start_sample:end_sample]

    # Need minimum duration for embedding
    min_samples = int(0.5 * sample_rate)  # 0.5 seconds minimum
    if len(segment) < min_samples:
        return None

    # Convert to tensor
    waveform = torch.tensor(segment).unsqueeze(0).float()

    # Extract embedding
    with torch.no_grad():
        embedding = model.encode_batch(waveform)

    return embedding.squeeze().cpu().numpy()


def compare_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        emb1: First embedding vector
        emb2: Second embedding vector

    Returns:
        Cosine similarity score between -1 and 1 (higher = more similar)
    """
    # Normalize vectors
    emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)

    # Compute cosine similarity
    return float(np.dot(emb1_norm, emb2_norm))


def match_speaker(
    embedding: np.ndarray,
    profiles: dict,
    threshold: float = 0.7
) -> tuple[Optional[str], float]:
    """
    Match an embedding against stored speaker profiles.

    Args:
        embedding: Speaker embedding to match
        profiles: Speaker profiles dict with embeddings
        threshold: Minimum similarity score to consider a match

    Returns:
        Tuple of (matched_speaker_id or None, best_similarity_score)
    """
    best_match = None
    best_score = -1.0

    for speaker_id, profile in profiles.get("profiles", {}).items():
        stored_embeddings = profile.get("embeddings", [])

        for stored_emb_b64 in stored_embeddings:
            stored_emb = embedding_from_base64(stored_emb_b64)
            score = compare_embeddings(embedding, stored_emb)

            if score > best_score:
                best_score = score
                if score >= threshold:
                    best_match = speaker_id

    return best_match, best_score


def embedding_to_base64(embedding: np.ndarray) -> str:
    """Convert embedding numpy array to base64 string for JSON storage."""
    return base64.b64encode(embedding.astype(np.float32).tobytes()).decode('utf-8')


def embedding_from_base64(b64_string: str) -> np.ndarray:
    """Convert base64 string back to embedding numpy array."""
    return np.frombuffer(base64.b64decode(b64_string), dtype=np.float32)
