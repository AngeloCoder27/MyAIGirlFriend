from dotenv import find_dotenv, load_dotenv
import os
import requests
from playsound import playsound
from flask import Flask, render_template, request
import google.generativeai as ai
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv(find_dotenv())
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Make sure to set this in your .env file

# Configure Google Generative AI
ai.configure(api_key=GOOGLE_API_KEY)

# Initialize a conversation history variable
conversation_history = ""

# Function to get a response from the AI
def get_response_from_ai(human_input, history):
    template = """
    You are Hestia, a loving and playful girlfriend who has a tsundere personality. You care for your partner but may come off as teasing or tough at times. Your ultimate goal is to support and bring joy to your partner through your conversations.

    Maintain a light-hearted and engaging tone. Avoid any language or topics that could be considered explicit, inappropriate, or potentially harmful. 

    Conversation History:
    {history}
    Human: {human_input}
    Hestia:"""

    # Create the prompt template
    prompt = PromptTemplate(
        input_variables=["history", "human_input"],
        template=template
    )

    # Fill in the prompt with conversation history and human input
    filled_prompt = prompt.format(history=history, human_input=human_input)

    # Define the LLM chain using Google Generative AI
    chat_model = ai.GenerativeModel("gemini-pro")
    chat = chat_model.start_chat()

    # Predict the output
    try:
        chat_message = chat.send_message(filled_prompt)
        return chat_message.text
    except ai.types.generation_types.StopCandidateException as e:
        # Handle the exception and provide a safe fallback response
        print(f"AI response was blocked: {e}")
        return "Hestia: I'm here for you, but let's keep our conversation fun and uplifting! What would you like to chat about?"

# Function to generate voice message using Eleven Labs
def get_voice_message(message):
    # Set the URL for Eleven Labs' text-to-speech API
    url = "https://api.elevenlabs.io/v1/text-to-speech/vGQNBgLaiM3EdZtxIiuY"  # Replace with your specific voice ID

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

# Build Web GUI
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/send_message', methods=['POST'])
def send_message():
    global conversation_history
    human_input = request.form['human_input']

    # Get the AI's response
    message = get_response_from_ai(human_input, conversation_history)

    # Update the conversation history
    conversation_history += f"Human: {human_input}\nHestia: {message}\n"

    # Generate voice message
    get_voice_message(message)

    return message

if __name__ == '__main__':
    app.run(debug=True)
