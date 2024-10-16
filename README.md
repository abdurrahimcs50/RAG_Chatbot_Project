
# RAG-Enhanced Chatbot Application 📚🤖

![RAG Chatbot Banner](assets/banner.png)

## Table of Contents

- [RAG-Enhanced Chatbot Application 📚🤖](#rag-enhanced-chatbot-application-)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Demo](#demo)
    - [Chat Interface](#chat-interface)
  - [Technologies Used](#technologies-used)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
  - [Usage](#usage)
  - [Logging](#logging)
  - [Screenshots](#screenshots)
    - [**Home Screen**](#home-screen)
    - [**API Key Input**](#api-key-input)
    - [**Model Selection**](#model-selection)
    - [**RAG Source Uploads**](#rag-source-uploads)
    - [**Chat Interface**](#chat-interface-1)
  - [Demo Video](#demo-video)
  - [Contributing](#contributing)
    - [**How to Contribute**](#how-to-contribute)
    - [**Guidelines**](#guidelines)
  - [License](#license)
  - [Contact](#contact)

## Introduction

Welcome to the **RAG-Enhanced Chatbot Application**, a powerful and scalable chatbot solution that leverages Retrieval-Augmented Generation (RAG) techniques to provide intelligent and context-aware responses. Built with Streamlit, Python, and advanced language models from OpenAI, this application is designed to enhance user interactions by integrating document and web-based knowledge sources.

Whether you're developing an e-commerce platform, a real estate service, or any application that requires dynamic and informed conversational agents, our chatbot offers the flexibility and robustness you need.

## Features

- **OpenAI GPT-4 Integration**: Utilizes OpenAI's GPT-4 models for advanced language generation.
- **RAG Integration**: Enhance chatbot responses by uploading documents or providing URLs, enabling the chatbot to retrieve and utilize external knowledge.
- **User-Friendly Interface**: Intuitive Streamlit-based UI with sidebar controls for API key management, model selection, and RAG source uploads.
- **Comprehensive Logging**: Detailed logging of user interactions, model selections, and system events stored in a dedicated `logs/` folder with log rotation.
- **Real-Time Date & Time**: Displays the current date and time for better context and user experience.
- **Responsive Design**: Optimized for various devices ensuring a seamless user experience.
- **Secure API Key Handling**: Secure input fields for API keys with sensitive information protection.

## Demo

Check out a short demo of the application in action:
### Chat Interface
![Chat Interface](https://github.com/abdurrahimcs50/RAG_Chatbot_Project/blob/main/assets/demo-thumbnail.png.png?raw=true)

<!-- [![Watch the Demo](https://img.youtube.com/vi/P8tOjiYEFqU/0.jpg)](https://www.youtube.com/watch?v=P8tOjiYEFqU) -->
[![Watch the Demo](https://github.com/abdurrahimcs50/RAG_Chatbot_Project/blob/main/assets/demo-thumbnail.png.png?raw=true)](https://www.youtube.com/watch?v=P8tOjiYEFqU)

*Click the image above to watch the demo video.*

## Technologies Used

- **Streamlit**: For building the interactive web application.
- **Python**: Core programming language.
- **LangChain**: For integrating language models.
- **OpenAI GPT-4**: For advanced language generation.
- **SQLite**: For lightweight database management.
- **Logging**: Python's built-in logging module for comprehensive logging.
- **Docker**: For containerized deployments (if applicable).
- **Other Libraries**: `dotenv`, `uuid`, etc.

## Installation

Follow these steps to set up the RAG-Enhanced Chatbot Application on your local machine.

### Prerequisites

- **Python 3.7+**
- **pip** (Python package installer)
- **Git** (for cloning the repository)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/rag-chatbot.git
   cd rag-chatbot
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**

   Create a `.env` file in the root directory and add your API keys:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   AZ_OPENAI_ENDPOINT=your_azure_endpoint
   ```

   *Ensure that you replace the placeholder values with your actual API keys.*

5. **Run the Application**

   ```bash
   streamlit run app.py
   ```

6. **Access the App**

   Open your web browser and navigate to `http://localhost:8501` to interact with the chatbot.

## Usage

1. **API Key Management**

   - Navigate to the sidebar to enter your OpenAI API keys.
   - These keys are essential for authenticating and utilizing the OpenAI GPT-4 models.

2. **Model Selection**

   - Choose your preferred language model from the dropdown menu in the sidebar.
   - Options include `openai/gpt-4o`, `openai/gpt-4o-mini`, and `azure-openai/gpt-4o` depending on your API keys.

3. **RAG Source Uploads**

   - **Upload Documents**: Click on the "Upload Documents for RAG Processing" button to upload PDFs, TXT, DOCX, or MD files.
   - **Add URLs**: Enter a URL to integrate web-based content into the chatbot's knowledge base.

4. **Chat Interface**

   - Type your message in the chat input field and press Enter.
   - The assistant will respond based on your input and the integrated RAG sources.

5. **Logging**

   - All interactions and system events are logged in the `logs/` directory.
   - Logs are rotated to prevent excessive file sizes, ensuring efficient storage management.

6. **Clear Chat**

   - Use the "Clear Chat" button in the sidebar to reset the conversation history.

## Logging

The application employs Python's built-in `logging` module to capture and store logs systematically.

- **Log Directory**: All logs are stored in the `logs/` folder.
- **Log Rotation**: Logs are rotated after reaching 5 MB, with up to 5 backup logs maintained to prevent storage issues.
- **Log Contents**:
  - Session initializations
  - API key inputs (without exposing the keys)
  - Model selections
  - RAG source uploads
  - User messages and assistant responses
  - Error and warning messages

*Example log entry:*

```
2024-10-16 14:30:45,123 - INFO - Initialized new session with ID: 123e4567-e89b-12d3-a456-426614174000
2024-10-16 14:31:10,456 - INFO - OpenAI API Key provided by user.
2024-10-16 14:31:15,789 - INFO - Selected model: openai/gpt-4o
2024-10-16 14:32:00,012 - INFO - 2 document(s) uploaded for RAG processing.
2024-10-16 14:32:30,345 - INFO - User input: How can I integrate RAG into my project?
2024-10-16 14:32:45,678 - INFO - Assistant response: To integrate RAG into your project...
```

## Screenshots

### **Home Screen**

![Home Screen](assets/home_screen.png)

### **API Key Input**

![API Key Input](assets/api_key_input.png)

### **Model Selection**

![Model Selection](assets/model_selection.png)

### **RAG Source Uploads**

![RAG Source Uploads](assets/rag_uploads.png)

### **Chat Interface**

![Chat Interface](assets/chat_interface.png)

*Replace the image paths (`assets/*.png`) with the actual paths to your images.*

## Demo Video

Watch the application in action:

[![Watch the Demo](assets/demo-thumbnail.png)](https://www.youtube.com/watch?v=your-demo-video-link)

*Click the image above to watch the demo video.*

## Contributing

We welcome contributions from the community! Whether it's bug fixes, feature enhancements, or documentation improvements, your input is valuable.

### **How to Contribute**

1. **Fork the Repository**

   Click the "Fork" button at the top right of the repository page.

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/yourusername/rag-chatbot.git
   cd rag-chatbot
   ```

3. **Create a New Branch**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

4. **Make Your Changes**

   Implement your feature or bug fix.

5. **Commit Your Changes**

   ```bash
   git commit -m "Add your descriptive commit message"
   ```

6. **Push to Your Fork**

   ```bash
   git push origin feature/YourFeatureName
   ```

7. **Open a Pull Request**

   Navigate to the original repository and click "Compare & pull request."

### **Guidelines**

- **Code Quality**: Ensure your code follows best practices and is well-documented.
- **Testing**: If applicable, include tests to verify your changes.
- **Documentation**: Update the README or other documentation if your changes affect usage.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions, suggestions, or feedback, feel free to reach out:

- **MD Abdur Rahim (Kawsar)**
- **Email**: [rahim@example.com](mailto:rahim@example.com)
- **Website**: [www.rahim.com.bd](https://www.rahim.com.bd/)
- **LinkedIn**: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)

---

© 2021 - 2024 [RahimTech](https://www.rahim.com.bd/). All rights reserved. Developed by [**MD Abdur Rahim**](https://www.rahim.com.bd/).