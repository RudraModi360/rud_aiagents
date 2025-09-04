import json
import numpy as np
import gradio as gr
import sounddevice as sd  # better low-latency audio playback
from groq import Groq
from fastrtc import get_tts_model

tts_model = get_tts_model()
groq = Groq()

conversations: dict[str, list[dict[str, str]]] = {}


def play_audio_stream(text: str):
    """Stream TTS audio directly to speaker using sounddevice."""
    stream = None
    samplerate = None

    try:
        for rate, chunk in tts_model.stream_tts_sync(text):
            if stream is None:  # Open stream on first chunk
                samplerate = rate
                stream = sd.OutputStream(samplerate=rate, channels=1, dtype="float32")
                stream.start()

            # Ensure float32 (sounddevice expects float32 in -1..1)
            if chunk.dtype != np.float32:
                audio = chunk.astype(np.float32)
            else:
                audio = chunk

            stream.write(audio)

    finally:
        if stream:
            stream.stop()
            stream.close()


def chat_stream(user_text, history, session_id="default"):
    """Chat + background TTS playback."""
    if session_id not in conversations:
        conversations[session_id] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that can answer questions and help with tasks."
                    ' Please return a short (for TTS) and long (for chatbot) response in JSON format:\n\n'
                    '{"short": "...", "long": "..."}'
                ),
            }
        ]

    messages = conversations[session_id]
    messages.append({"role": "user", "content": user_text})

    completion = groq.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        response_format={
            "type": "json_object",
            "attributes": [
                {"name": "short", "type": "string", "description": "Short TTS-friendly answer"},
                {"name": "long", "type": "string", "description": "Detailed long answer for chat UI"}
            ]
        }
    )

    response = json.loads(completion.choices[0].message.content)
    short_response = response["short"]
    long_response = response["long"]

    messages.append({"role": "assistant", "content": long_response})
    conversations[session_id] = messages

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": long_response})

    # 🔊 Play TTS in background (non-blocking)
    import threading
    threading.Thread(target=play_audio_stream, args=(short_response,), daemon=True).start()

    return history


# Gradio UI
with gr.Blocks(css="""
    .gr-chatbot {border-radius: 12px; border: 1px solid #e0e0e0; padding: 10px;}
    .gr-textbox {border-radius: 12px; border: 1px solid #e0e0e0;}
""") as demo:
    with gr.Column():
        chatbot = gr.Chatbot(label="Assistant", type="messages")
        txt = gr.Textbox(label="Type your message", placeholder="Type a message and press Enter")

    txt.submit(
        chat_stream,
        [txt, chatbot],
        [chatbot],
        queue=True
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, debug=True)
