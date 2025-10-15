import os
import requests
import pyttsx3
from groq import Groq

# ---------------------------
# Setup
# ---------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tts = pyttsx3.init()

def speak(text: str):
    print("Agent:", text)
    tts.say(text)
    tts.runAndWait()

# ---------------------------
# Tool definitions
# ---------------------------
def get_weather(city: str):
    """Fetch simple weather string using wttr.in"""
    resp = requests.get(f"https://wttr.in/{city}?format=3")
    return resp.text

def get_time(city: str):
    """Fetch current time using worldtimeapi.org"""
    resp = requests.get(f"http://worldtimeapi.org/api/timezone/{city}")
    if resp.status_code == 200:
        data = resp.json()
        return data["datetime"]
    return "Could not fetch time."

tools = {
    "get_weather": get_weather,
    "get_time": get_time
}

# ---------------------------
# Tool schema (Groq function calling)
# ---------------------------
functions = [
    {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "get_time",
        "description": "Get the current time in a city (use TZ format like Europe/London)",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "Timezone string"}
            },
            "required": ["city"]
        }
    }
]

# ---------------------------
# Main loop
# ---------------------------
while True:
    user_input = input("You: ")

    # Ask Groq agent
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # one of Groq’s fast models
        messages=[{"role": "user", "content": user_input}],
        functions=functions,
        function_call="auto"
    )

    msg = response.choices[0].message
    print("Response : ",msg)
    # If tool call requested
    if msg.function_call:
        fn_name = msg.function_call.name
        args = eval(msg.function_call.arguments)
        print("Args : ",args)
        # Narrate before calling
        speak(f"Alright, let me check {args['city']} using {fn_name}.")

        # Run the tool
        result = tools[fn_name](**args)

        # Feed result back into Groq for natural summary
        followup = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Summarize results naturally for spoken voice."},
                {"role": "user", "content": f"Tool {fn_name} returned: {result}"}
            ]
        )
        speak(followup.choices[0].message.content)

    else:
        # Direct response (no tool)
        speak(msg.content)
