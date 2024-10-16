import streamlit as st
import os
import dotenv
import uuid
import logging
from logging.handlers import RotatingFileHandler
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from rag_methods import (
    load_doc_to_db,
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
)

# --- Logging Configuration ---
LOG_DIR = "logs"
LOG_FILE = "rag_chatbot.log"

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Set up rotating file handler
file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, LOG_FILE),
    maxBytes=5*1024*1024,  # 5 MB
    backupCount=5
)

# Set up logging format
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        file_handler,            # Log to file with rotation
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

# Define available models based on environment variables
if "AZ_OPENAI_API_KEY" not in os.environ:
    MODELS = [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "anthropic/claude-3-5-sonnet-20240620",
    ]
else:
    MODELS = ["azure-openai/gpt-4o"]

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="RAG-Enhanced Chatbot Application",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Header ---
st.html(
    """<h2 style="text-align: center;">📚🔍 <i> Elevate Your LLM with RAG-Powered Insights </i> 🤖💬</h2>"""
)
logger.info("Application header rendered.")

# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logger.info(f"Initialized new session with ID: {st.session_state.session_id}")

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []
    logger.info("Initialized RAG sources list.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"},
    ]
    logger.info("Initialized default conversation messages.")

# --- Sidebar with Notes and API Key Inputs ---
with st.sidebar:
    # --- Sidebar Header ---
    st.header("📌 Notes")
    st.write(
        """
        - **API Keys**: Ensure you have valid API keys for the selected LLM model.
        - **RAG Sources**: Upload documents or enter URLs to enhance your chatbot's responses.
        - **Model Selection**: Choose the model that best fits your needs from the dropdown.
        """
    )
    logger.info("Sidebar notes section rendered.")

    # Check for Azure OpenAI API Key
    if "AZ_OPENAI_API_KEY" not in os.environ:
        # OpenAI API Key Input
        default_openai_api_key = os.getenv("OPENAI_API_KEY", "")
        with st.expander("🔐 OpenAI API Key"):
            openai_api_key = st.text_input(
                "Enter your OpenAI API Key (https://platform.openai.com/) to get started",
                value=default_openai_api_key,
                type="password",
                key="openai_api_key",
            )
            if openai_api_key:
                logger.info("OpenAI API Key provided by user.")

# --- Main Content ---
# Check if necessary API keys are present
missing_openai = (
    openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key
)
# if missing_openai and ("AZ_OPENAI_API_KEY" not in os.environ):
if missing_openai and ("AZ_OPENAI_API_KEY" not in os.environ):
    st.write("#")
    st.warning("⚠️ Please enter an API Key to unlock model access...")
    logger.warning("No valid API keys detected for OpenAI or Azure OpenAI models.")
else:
    # Sidebar for model selection and RAG options
    with st.sidebar:
        st.divider()
        logger.info("Sidebar model selection and RAG options rendered.")
        
        # Model Selection
        models = []
        for model in MODELS:
            if "openai" in model and not missing_openai:
                models.append(model)
        
        selected_model = st.selectbox(
            "🤖 Choose Your LLM Model",
            options=models,
            key="model",
        )
        logger.info(f"Selected model: {selected_model}")

        # RAG Toggle and Clear Chat Button
        cols0 = st.columns(2)
        with cols0[0]:
            is_vector_db_loaded = (
                "vector_db" in st.session_state
                and st.session_state.vector_db is not None
            )
            st.toggle(
                "Use RAG",
                value=is_vector_db_loaded,
                key="use_rag",
                disabled=not is_vector_db_loaded,
            )
            logger.info(f"RAG usage toggled: {'Enabled' if is_vector_db_loaded else 'Disabled'}")

        with cols0[1]:
            if st.button(
                "Clear Chat",
                on_click=lambda: st.session_state.messages.clear(),
                type="primary",
            ):
                logger.info("Chat messages cleared by user.")

        # Active RAG Sources
        st.header("🗂️ Active RAG Sources")

        # File Uploader for RAG Documents
        uploaded_files = st.file_uploader(
            "📄 Upload Documents for RAG Processing",
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
            on_change=load_doc_to_db,
            key="rag_docs",
        )
        if uploaded_files:
            logger.info(f"{len(uploaded_files)} document(s) uploaded for RAG processing.")

        # URL Input for RAG
        rag_url = st.text_input(
            "🌐 Add a URL for Web-Based RAG",
            placeholder="https://example.com",
            on_change=load_url_to_db,
            key="rag_url",
        )
        if rag_url:
            logger.info(f"URL added for RAG: {rag_url}")

        # Display Loaded Documents
        with st.expander(
            f"📚 Documents in DB ({len(st.session_state.rag_sources)})"
        ):
            st.write(
                [] if not is_vector_db_loaded else st.session_state.rag_sources
            )
            if is_vector_db_loaded:
                logger.info(f"Displaying {len(st.session_state.rag_sources)} RAG source(s) in the database.")

    # Initialize LLM Stream based on selected model
    model_provider = selected_model.split("/")[0]
    llm_stream = None

    if model_provider == "openai":
        llm_stream = ChatOpenAI(
            api_key=openai_api_key,
            model_name=selected_model.split("/")[-1],
            temperature=0.3,
            streaming=True,
        )
        logger.info("Initialized OpenAI ChatOpenAI stream.")

    # Display Chat Messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            logger.info(f"{message['role'].capitalize()}: {message['content']}")

    # Capture User Input and Generate Responses
    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        logger.info(f"User input: {prompt}")

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Prepare messages for LLM
            messages = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages
            ]

            try:
                if not st.session_state.use_rag:
                    response_generator = stream_llm_response(llm_stream, messages)
                    logger.info("Generating response without RAG.")
                else:
                    response_generator = stream_llm_rag_response(llm_stream, messages)
                    logger.info("Generating RAG-enhanced response.")

                # Stream and display the response
                for chunk in response_generator:
                    if chunk:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                logger.info(f"Assistant response: {full_response}")

            except Exception as e:
                logger.error(f"Error during response generation: {e}")
                message_placeholder.markdown("⚠️ An error occurred while generating the response.")

# --- Footer ---
with st.sidebar:
    st.divider()
    # Add copyright text with links and styling
    st.markdown(
        """
        <div style='text-align: center; font-size: 12px; color: gray;'>
            © 2021 - 2024 <a href='https://www.rahim.com.bd/' target='_blank' style='text-decoration: none; color: #1E90FF;'>
            <strong>RahimTech</strong></a>. All rights reserved.<br>
            Developed by <a href='https://www.rahim.com.bd/' target='_blank' style='text-decoration: none; color: #1E90FF;'>
            <strong>MD Abdur Rahim</strong></a>
        </div>
        """,
        unsafe_allow_html=True
    )
    logger.info("Footer rendered.")
