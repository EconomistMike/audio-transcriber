"""
Speaker Management Page

Enroll, view, edit, and delete speaker profiles.
"""

import tempfile
from pathlib import Path

import streamlit as st

from transcriber import (
    list_speakers,
    load_profiles,
    save_profiles,
    enroll_speaker,
)
from transcriber.speakers import remove_speaker

st.set_page_config(page_title="Speakers", page_icon="👥", layout="wide")

st.title("👥 Speaker Management")
st.markdown("Enroll speakers for automatic voice recognition.")

st.divider()

# Enroll new speaker section
st.subheader("Enroll New Speaker")

with st.form("enroll_form"):
    col1, col2 = st.columns([1, 2])

    with col1:
        speaker_name = st.text_input(
            "Speaker Name",
            placeholder="e.g., John Smith"
        )

    with col2:
        audio_file = st.file_uploader(
            "Voice Sample",
            type=["mp3", "wav", "m4a", "ogg", "flac", "webm"],
            help="Upload an audio file of the speaker talking (at least 5 seconds recommended)"
        )

    submitted = st.form_submit_button("Enroll Speaker", use_container_width=True)

    if submitted:
        if not speaker_name:
            st.error("Please enter a speaker name.")
        elif not audio_file:
            st.error("Please upload an audio file.")
        else:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(audio_file.name).suffix
            ) as tmp:
                tmp.write(audio_file.read())
                tmp_path = Path(tmp.name)

            try:
                with st.spinner(f"Enrolling {speaker_name}..."):
                    enroll_speaker(speaker_name, tmp_path)
                st.success(f"Successfully enrolled **{speaker_name}**!")
                st.rerun()
            except Exception as e:
                st.error(f"Error enrolling speaker: {e}")
            finally:
                tmp_path.unlink(missing_ok=True)

st.divider()

# List enrolled speakers
st.subheader("Enrolled Speakers")

speakers = list_speakers()

if not speakers:
    st.info("No speakers enrolled yet. Use the form above to add voice samples.")
else:
    for speaker in speakers:
        with st.expander(f"**{speaker['name']}** ({speaker['embedding_count']} sample(s))", expanded=False):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**ID:** `{speaker['id']}`")
                st.markdown(f"**Created:** {speaker['created'][:10]}")
                st.markdown(f"**Samples:** {speaker['embedding_count']}")

                if speaker.get("sample_files"):
                    st.markdown("**Sample files:**")
                    for f in speaker["sample_files"]:
                        st.markdown(f"- {f}")

            with col2:
                # Add more samples
                st.markdown("**Add Sample**")
                more_audio = st.file_uploader(
                    "Upload",
                    type=["mp3", "wav", "m4a", "ogg", "flac", "webm"],
                    key=f"add_{speaker['id']}",
                    label_visibility="collapsed"
                )

                if more_audio:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=Path(more_audio.name).suffix
                    ) as tmp:
                        tmp.write(more_audio.read())
                        tmp_path = Path(tmp.name)

                    try:
                        with st.spinner("Adding sample..."):
                            enroll_speaker(speaker["name"], tmp_path)
                        st.success("Sample added!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        tmp_path.unlink(missing_ok=True)

                st.markdown("---")

                # Delete speaker
                if st.button("🗑️ Delete", key=f"del_{speaker['id']}", type="secondary"):
                    st.session_state[f"confirm_delete_{speaker['id']}"] = True

                if st.session_state.get(f"confirm_delete_{speaker['id']}"):
                    st.warning("Are you sure?")
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("Yes", key=f"yes_{speaker['id']}"):
                            remove_speaker(speaker["id"])
                            st.success(f"Deleted {speaker['name']}")
                            st.rerun()
                    with col_no:
                        if st.button("No", key=f"no_{speaker['id']}"):
                            st.session_state[f"confirm_delete_{speaker['id']}"] = False
                            st.rerun()
