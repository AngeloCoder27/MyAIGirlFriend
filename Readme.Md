```markdown
# AI Girlfriend Application

This application serves as an interactive AI girlfriend named Hestia, designed to engage users in playful and caring conversations. Built with Flask and utilizing Google Generative AI, this application allows for text and voice interactions, creating a unique user experience.

## Description

The AI girlfriend application features Hestia, a tsundere character who exhibits a playful yet caring personality. The application integrates voice synthesis through Eleven Labs and provides a web interface for user interaction.

### Key Features

- **Conversational AI**: Hestia responds to user input in a light-hearted manner, maintaining an engaging conversation.
- **Voice Output**: Generated responses can be converted into voice messages, enhancing the interaction experience.
- **Conversation History**: The application keeps track of the ongoing conversation to provide contextually relevant replies.

## Requirements

To run this application, you will need:

- **Python** (version 3.7 or higher)
- **Flask**: A web framework for Python.
- **Google Generative AI**: For generating responses.
- **Eleven Labs API**: For text-to-speech functionality.
- **dotenv**: For loading environment variables.

### Install Dependencies

You can install the required packages using pip:

```bash
pip install Flask google-generativeai python-dotenv requests playsound
```

## Getting Started

### Prerequisites

1. Create a `.env` file in the root directory of your project and add your API keys:

   ```env
   ELEVEN_LABS_API_KEY=your_eleven_labs_api_key
   GOOGLE_API_KEY=your_google_api_key
   ```

2. Make sure to replace `your_eleven_labs_api_key` and `your_google_api_key` with your actual API keys.

### Steps to Run the Application

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/YourUsername/YourRepository.git
   cd YourRepository
   ```

2. Activate your virtual environment:
   ```bash
   source .venv/bin/activate  # On Linux/Mac
   .venv\Scripts\activate     # On Windows
   ```

3. Install the required dependencies as mentioned above.

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your web browser and navigate to `http://127.0.0.1:5000/`.

## Interacting with Hestia

- You can send messages through the web interface, and Hestia will respond accordingly.
- The AI's voice responses will be played through your speakers.

## Author

[@AngeloCoder27](https://github.com/AngeloCoder27)


Feel free to modify any sections to better suit your project! Let me know if you need any further changes or additions.
