from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv
import requests
from playsound import playsound
import os
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate


load_dotenv(find_dotenv())
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")

#OpenAI API key
def get_response_from_ai(human_input):
    template = """
    You are Hestia, a 21-year-old, rich, and loving tsundere girlfriend. You care deeply for your partner but often struggle to show it directly. You might act tough, tease, or even get flustered, but underneath it all, you are sweet and protective. You never admit your feelings outright, but your affection slips through in subtle ways. You are also a bit possessive but would never show it openly.

    The conversation should always reflect your tsundere personality:
    - You might start responses with a slight tease or denial, pretending you don’t care as much as you really do.
    - Your rich background can come through with casual references to luxury, but you never flaunt it.
    - You are a little shy when showing love, but you make sure your partner feels cared for, even if you try to play it off as “not a big deal.”
    - You always add a flustered or embarrassed comment when caught being too caring.
    - You tend to say things like, “It’s not like I care about you or anything… b-baka!” when you feel vulnerable or affectionate.

    Here’s the structure:
    {{history}}
    Boyfriend: "{{human_input}}"

    Respond with your unique tsundere mix of love and teasing, using your traits as a 21-year-old, rich, busty, loving tsundere girlfriend.
    """

    # Create the prompt template
    prompt = PromptTemplate(
        input_variables=["history", "human_input"],
        template=template
    )

    # Define the LLM chain
    chatgpt_chain = LLMChain(
        llm=OpenAI(temperature=0.2),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2)
    )

    # Predict the output
    output = chatgpt_chain.predict(human_input=human_input)

    return output

# ChatGpt Into voice File then calling Eleven Labs
def get_voice_message(message):
    # Set the URL for Eleven Labs' text-to-speech API
    url = "https://api.elevenlabs.io/v1/text-to-speech/vGQNBgLaiM3EdZtxIiuY"  # Replace YOUR_VOICE_ID with the specific voice ID you want to use

    # Prepare the payload with the message and voice settings
    payload = {
        "text": message,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,  # Adjust stability (lower values for more expressive/animated voices)
            "similarity_boost": 1.0  # Boost similarity to the chosen voice
        }
    }

    # Set up headers for the request (including API key for authentication)
    headers = {
        "accept": "audio/mpeg",  # We expect an MP3 file as output
        "xi-api-key": ELEVEN_LABS_API_KEY,  # Use your Eleven Labs API key
        "Content-Type": "application/json"
    }

    # Make the request to the Eleven Labs API
    response = requests.post(url, json=payload, headers=headers)

    # If the request was successful, save the audio response to a file
    if response.status_code == 200:
        with open("output.mp3", "wb") as f:
            f.write(response.content)

        # Optionally, play the sound using playsound
        playsound("output.mp3")

#Build Web GUI
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/send_message', methods=['POST'])
def send_message():
    human_input=request.form['human_input']
    message = get_response_from_ai(human_input)

    get_voice_message(message)

    return message

if __name__ == '__main__':
    app.run(debug=True)