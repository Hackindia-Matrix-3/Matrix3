from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import base64

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Streamlit app
st.set_page_config(page_title="Agriculture RAG Chatbot")
st.header("🌱 AI-Powered Agriculture Q&A Chatbot with Voice")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "question_input" not in st.session_state:
    st.session_state.question_input = ""

# Load and process agricultural knowledge base from a PDF
@st.cache_resource
def load_knowledge_base():
    try:
        pdf_path = "data.pdf"
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split text into smaller chunks
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        
        # Extract text content
        text_chunks = [doc.page_content for doc in texts]
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(text_chunks)
        
        return {
            "vectorizer": vectorizer,
            "tfidf_matrix": tfidf_matrix,
            "text_chunks": text_chunks
        }
    except Exception as e:
        st.error(f"Error loading knowledge base: {str(e)}")
        return None

# Initialize knowledge base
kb_data = None
with st.spinner("Loading knowledge base..."):
    try:
        kb_data = load_knowledge_base()
        if kb_data:
            st.success("Knowledge base loaded successfully!")
    except Exception as e:
        st.error(f"Failed to initialize knowledge base: {str(e)}")

# Function to retrieve relevant documents
def get_relevant_documents(question, top_k=5):
    if kb_data is None:
        return []
    
    vectorizer = kb_data["vectorizer"]
    tfidf_matrix = kb_data["tfidf_matrix"]
    text_chunks = kb_data["text_chunks"]
    
    # Transform query to TF-IDF vector
    question_vector = vectorizer.transform([question])
    
    # Calculate cosine similarity between query and documents
    similarities = cosine_similarity(question_vector, tfidf_matrix).flatten()
    
    # Get indices of top-k most similar documents
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # Return relevant documents
    relevant_docs = [text_chunks[i] for i in top_indices]
    return relevant_docs

# Function to retrieve data & generate responses
def get_rag_response(question):
    if kb_data is None:
        return "Knowledge base could not be loaded. Please check your PDF file and configuration."
    
    try:
        # Get relevant documents
        relevant_docs = get_relevant_documents(question)
        
        if not relevant_docs:
            return "No relevant information found in the knowledge base."

        # Combine retrieved data into a prompt with chat history
        chat_history_str = "\n".join(st.session_state.chat_history[-5:])  # Keep last 5 messages for context
        context = "\n".join(relevant_docs)
        prompt = f"""You are an agricultural expert assistant. Use ONLY the following agricultural 
                  knowledge to answer the user's question. If the information isn't in the provided 
                  context, say that you don't have that information.
                  
                  Previous Conversation:
                  {chat_history_str}
                  
                  Context:
                  {context}
                  
                  Question: {question}
                  Answer:"""
        
        # Generate response using Gemini AI
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Voice recognition function
def listen_for_speech():
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening... Speak now!")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)
            
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand what you said."
        except sr.RequestError:
            return "Sorry, my speech service is down."
    except Exception as e:
        return f"Error with speech recognition: {str(e)}"

# Text-to-speech function
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        # Create HTML audio element with autoplay
        audio_bytes = fp.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_tag = f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_base64}"></audio>'
        
        return audio_tag
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None

# Callback for voice input
def set_speech_text():
    speech_text = listen_for_speech()
    if speech_text and not speech_text.startswith("Sorry"):
        st.session_state.question_input = speech_text
    else:
        st.warning(speech_text)

# Callback for example questions
def set_example_question(question):
    st.session_state.question_input = question

# Streamlit UI
st.subheader("Ask your question")
col1, col2 = st.columns([4, 1])

with col1:
    # Use the session state value as the default value for the text input
    input_question = st.text_input("🌾 Ask an agricultural question:", 
                                  value=st.session_state.question_input,
                                  key="text_input")
    submit_text = st.button("🔍 Get Answer")

with col2:
    st.write(" ")
    st.write(" ")
    voice_button = st.button("🎤 Voice Input", on_click=set_speech_text)

# Add toggle for text-to-speech responses
tts_enabled = st.sidebar.toggle("Enable Voice Responses", value=True)

# Handle user input (from text or voice)
if submit_text and input_question:
    with st.spinner("Generating answer..."):
        response = get_rag_response(input_question)
        
        # Store question and response in chat history
        st.session_state.chat_history.append(f"**User:** {input_question}")
        st.session_state.chat_history.append(f"**AI:** {response}")

        # Display response
        st.subheader("🤖 AI Response:")
        st.markdown(response)
        
        # Convert response to speech if enabled
        if tts_enabled:
            audio_html = text_to_speech(response)
            if audio_html:
                st.markdown(audio_html, unsafe_allow_html=True)
        
        # Clear the input after processing
        st.session_state.question_input = ""
                
elif submit_text and not input_question:
    st.warning("Please enter a question first.")

# Display chat history
st.subheader("📝 Chat History")
for chat in st.session_state.chat_history:
    st.markdown(chat)

# Display PDF info
st.sidebar.header("About")
st.sidebar.info("This chatbot answers questions based on agricultural knowledge from 'farmerbook.pdf'.")

# Add example questions
st.sidebar.header("Example Questions")
example_questions = [
    "What are the best crops to grow in summer?",
    "How to treat common plant diseases?",
    "What are sustainable farming practices?",
    "How to improve soil fertility naturally?"
]
for i, q in enumerate(example_questions):
    # Use a callback to set the question text
    if st.sidebar.button(q, key=f"example_{i}"):
        set_example_question(q)

# Add voice assistant instructions
st.sidebar.header("Voice Assistant")
st.sidebar.info("""
- Click '🎤 Voice Input' to ask questions verbally
- Toggle 'Enable Voice Responses' to hear AI responses
- Make sure your browser has microphone permissions enabled
""")

# Requirements instruction
st.sidebar.header("Setup Requirements")
st.sidebar.code("""
pip install streamlit google-generativeai langchain langchain-community PyMuPDF scikit-learn python-dotenv SpeechRecognition gTTS
""")