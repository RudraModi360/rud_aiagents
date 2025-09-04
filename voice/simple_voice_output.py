import os
import time
import threading
import numpy as np
import sounddevice as sd
import gradio as gr
from fastrtc import ReplyOnPause, Stream, get_tts_model, get_stt_model, AlgoOptions
from groq import Groq
import json

# Init models
stt_model = get_stt_model()
tts_model = get_tts_model()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Conversations memory
conversations: dict[str, list[dict[str, str]]] = {}


def play_audio_stream(text: str):
    """Play TTS stream directly to speaker."""
    stream = None
    try:
        for rate, chunk in tts_model.stream_tts_sync(text):
            if stream is None:
                stream = sd.OutputStream(samplerate=rate, channels=1, dtype="float32")
                stream.start()
            audio = chunk.astype(np.float32) if chunk.dtype != np.float32 else chunk
            stream.write(audio)
    finally:
        if stream:
            stream.stop()
            stream.close()


def get_llm_response(user_text: str, session_id="default") -> tuple[str, str]:
    """Query Groq LLM and return (short, long)."""
    if session_id not in conversations:
        conversations[session_id] = [
            {
                "role": "system",
                "content": (
                    "You are JARVIS, a warm, efficient, and expressive assistant. "
                    "Always return JSON with a short response for TTS and long response for chat:\n\n"
                    '{"short": "...", "long": "..."}'
                ),
            }
        ]

    messages = conversations[session_id]
    messages.append({"role": "user", "content": user_text})

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
    )

    raw_content = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw_content)
        short_resp, long_resp = parsed["short"], parsed["long"]
    except Exception:
        # fallback: at least keep it usable
        long_resp = raw_content
        short_resp = raw_content.split(".")[0]

    messages.append({"role": "assistant", "content": long_resp})
    conversations[session_id] = messages

    return short_resp, long_resp


def echo(audio):
    """Handles continuous mic input -> STT -> LLM -> TTS."""
    st_time = time.time()
    transcript = stt_model.stt(audio)
    if not transcript.strip():
        return  # ignore silence
    transcription_time = time.time()
    print(f"STT: {transcript} ({transcription_time - st_time:.2f}s)")

    short_resp, long_resp = get_llm_response(transcript)
    print("Assistant:", long_resp)

    # Log into chatbot session
    chatbot_state.append({"role": "user", "content": transcript})
    chatbot_state.append({"role": "assistant", "content": long_resp})

    # 🔊 Send short_resp to TTS stream (UI playback only)
    for rate, chunk in tts_model.stream_tts_sync(short_resp):
        yield (rate, chunk)


# VAD tuning
options = AlgoOptions(started_talking_threshold=0.3, speech_threshold=0.3)

# Persistent state for chatbot
chatbot_state = []

# Audio stream (continuous mic input)
stream = Stream(
    ReplyOnPause(echo, algo_options=options),
    modality="audio",
    mode="send-receive"
)

# Gradio UI
with stream.ui as demo:
    with gr.Column():
        chatbot = gr.Chatbot(label="Assistant", type="messages")
        txt = gr.Textbox(label="Ask with keyboard", placeholder="Type and press Enter")

    # Text input pipeline
    def handle_text(user_text, history):
        short_resp, long_resp = get_llm_response(user_text)
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": long_resp})
        threading.Thread(target=play_audio_stream, args=(short_resp,), daemon=True).start()
        return history

    txt.submit(handle_text, [txt, chatbot], [chatbot])

    # keep chatbot synced with voice input
    demo.load(lambda: chatbot_state, None, chatbot)

demo.launch(share=True)
