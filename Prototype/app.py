# streamlit_app.py
import streamlit as st
import numpy as np
import soundfile as sf
import time
import os
import uuid
from pydub import AudioSegment
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, AudioProcessorBase

from backend import transcribe_with_whisper_local, transcribe_with_openai_api

# -------------------------
st.set_page_config(page_title="Live Video -> Live Transcription", layout="wide")
st.title("Live video transcription prototype (Python stack)")
st.markdown("""
This demo captures your webcam + microphone and transcribes short chunks (near-real-time).
Choose backend: **local whisper** or **OpenAI API** (you need an API key).
""")

# Sidebar options
with st.sidebar:
    st.header("Options")
    backend = st.selectbox("Transcription backend", ["local_whisper", "openai_api"])
    language = st.text_input("Target language (ISO code, e.g. 'en', 'hi', or leave blank to auto-detect)", value="en")
    chunk_seconds = st.slider("Chunk length (seconds)", min_value=2, max_value=10, value=5)
    whisper_model = st.selectbox("Local Whisper model (if using local)", ["tiny", "base", "small", "medium", "large"])
    openai_model = st.text_input("OpenAI model name (if using OpenAI)", value="gpt-4o-transcribe")
    openai_key = st.text_input("OpenAI API key (optional — uses OPENAI_API_KEY env var if blank)", type="password")

# show live transcript area
st.subheader("Live transcript")
transcript_area = st.empty()

# Simple buffer to hold cumulative text
if "full_transcript" not in st.session_state:
    st.session_state["full_transcript"] = ""

# -------------------------
# Audio processor which receives audio frames from browser
class TranscribeAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = bytearray()
        self.last_chunk_time = time.time()
        self.sample_rate = 48000  # browser sample rate
        self.channels = 1

    def recv(self, frames, timestamp):
        """
        frames: list of numpy arrays shaped (n_samples, n_channels)
        We'll append frames and once we have chunk_seconds worth of audio -> write wav and transcribe.
        """
        # Convert frames to bytes and append
        for frame in frames:
            # frame is ndarray float32 in [-1,1], shape (n, channels)
            # convert to int16 PCM
            # downmix to mono
            if frame.ndim > 1:
                mono = np.mean(frame, axis=1)
            else:
                mono = frame
            # convert float32 -> int16
            pcm16 = (mono * 32767).astype(np.int16)
            self.buffer.extend(pcm16.tobytes())

        # check if buffer length >= chunk_seconds
        current_len_seconds = len(self.buffer) / (2 * self.sample_rate)  # 2 bytes per sample
        if current_len_seconds >= st.session_state.get("chunk_seconds", 5):
            # flush buffer to temp wav
            tmp_filename = f"tmp_{uuid.uuid4().hex}.wav"
            # write using soundfile
            data = np.frombuffer(self.buffer, dtype=np.int16).astype(np.float32) / 32767.0
            sf.write(tmp_filename, data, self.sample_rate)
            # reset buffer
            self.buffer = bytearray()

            # call transcription in background-ish: we can't spawn heavy background threads in streamlit safely,
            # so call synchronously (quick) — acceptable for prototype
            try:
                if st.session_state.get("backend") == "local_whisper":
                    res = transcribe_with_whisper_local(tmp_filename, language=st.session_state.get("language") or None, model_name=st.session_state.get("whisper_model","small"))
                else:
                    res = transcribe_with_openai_api(tmp_filename, language=st.session_state.get("language") or None, api_key=st.session_state.get("openai_key") or None, model=st.session_state.get("openai_model"))
                text = res.get("text","").strip()
            except Exception as e:
                text = f"[transcription error: {e}]"
            # append to session_state full transcript and update UI
            st.session_state["full_transcript"] += ("\n" + text) if text else ""
            transcript_area.markdown(st.session_state["full_transcript"].replace("\n","  \n"))
            print(text)
            # cleanup
            try:
                os.remove(tmp_filename)
            except:
                pass

        # For playback we return the frames unchanged
        return frames

# Save selected options into session_state for the audio processor to read
st.session_state["backend"] = backend
st.session_state["language"] = language.strip() if language else None
st.session_state["chunk_seconds"] = chunk_seconds
st.session_state["whisper_model"] = whisper_model
st.session_state["openai_model"] = openai_model
st.session_state["openai_key"] = openai_key

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Start the webrtc streamer; video + audio
webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": True},
    audio_processor_factory=TranscribeAudioProcessor,
    async_processing=False,
)

st.markdown("""
**Notes & limitations**
- This is a prototype. For production: use a dedicated WebRTC media server or a robust chunking/queueing pipeline.
- Local Whisper on CPU may be slow for low-latency; prefer small/tiny or use GPU.
- OpenAI API requires a valid API key and will incur costs.
""")
