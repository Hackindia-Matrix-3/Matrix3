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
from PIL import Image
import pickle  # For loading your pretrained model
import tensorflow as tf  # For loading TensorFlow/Keras models
import traceback

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Streamlit app
st.set_page_config(page_title="Agriculture RAG Chatbot")
st.header("🌱 AI-Powered Agriculture Q&A Chatbot with Voice & Image Analysis")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "text"  # Options: "text", "image", "disease_predict"

# Load pretrained disease prediction model
@st.cache_resource
def load_disease_model():
    try:
        # Replace with the path to your model file
        model_path = "plant_identification_model.h5"  # Change to your model path and format
        
        # Try loading the model based on file extension
        if model_path.endswith('.h5'):
            try:
                model = tf.keras.models.load_model(model_path)
                return model
            except Exception as e:
                st.error(f"Error loading H5 model: {str(e)}")
                st.error(traceback.format_exc())
        elif model_path.endswith('.pkl'):
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                return model
            except Exception as e:
                st.error(f"Error loading pickle model: {str(e)}")
                st.error(traceback.format_exc())
        elif "saved_model" in model_path:
            try:
                model = tf.saved_model.load(model_path)
                return model
            except Exception as e:
                st.error(f"Error loading TensorFlow SavedModel: {str(e)}")
                st.error(traceback.format_exc())
        else:
            st.error(f"Unsupported model format. Please use .h5, .pkl, or SavedModel format.")
            return None
    except Exception as e:
        st.error(f"Error loading disease prediction model: {str(e)}")
        st.error(traceback.format_exc())
        return None

# Function for disease prediction
def predict_disease(image):
    try:
        # Load the model (if not already loaded)
        model = load_disease_model()
        if model is None:
            return "Failed to load disease prediction model."
        
        # Preprocess the image for your model
        processed_image = preprocess_image_for_model(image)
        if processed_image is None:
            return "Failed to preprocess image for prediction."
        
        # Make prediction
        prediction = model.predict(processed_image)
        
        # Format the results in a user-friendly way
        result = format_prediction_results(prediction)
        
        return result
    except Exception as e:
        st.error(traceback.format_exc())
        return f"Error making disease prediction: {str(e)}"

# Helper function to preprocess image for your model
def preprocess_image_for_model(image):
    try:
        # Resize to expected dimensions (adjust to your model's requirements)
        img_resized = image.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(img_resized)
        
        # Normalize pixel values (adjust as needed)
        img_array = img_array / 255.0
        
        # Expand dimensions to match model input shape [batch_size, height, width, channels]
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        st.error(traceback.format_exc())
        return None

# # Helper function to format prediction results
# def format_prediction_results(prediction):
#     try:
#         # This function should be customized for your model's output format
#         # Example for a classification model:
        
#         # Get the highest probability class
#         predicted_class_index = np.argmax(prediction[0])
#         confidence = prediction[0][predicted_class_index] * 100
        
#         # Replace this with your actual class labels
#         class_labels = [
#             "Healthy",
#             "Disease 1",
#             "Disease 2",
#             # Add all your disease classes here
#         ]
        
#         # Check if we have enough class labels
#         if predicted_class_index >= len(class_labels):
#             predicted_disease = f"Unknown (Class {predicted_class_index})"
#         else:
#             predicted_disease = class_labels[predicted_class_index]
        
#         # Create formatted result
#         result = f"## Plant Disease Analysis\n\n"
#         result += f"### Prediction: {predicted_disease}\n"
#         result += f"### Confidence: {confidence:.2f}%\n\n"
        
#         # Add recommendations based on predicted disease
#         result += "### Recommended Actions:\n"
#         if predicted_disease == "Healthy":
#             result += "- Plant appears healthy\n"
#             result += "- Continue regular maintenance\n"
#             result += "- Monitor for any changes\n"
#         else:
#             result += "- Consider treatment options for the identified condition\n"
#             result += "- Isolate affected plants to prevent spread\n"
#             result += "- Consult with a plant specialist for specific treatment\n"
        
#         return result
#     except Exception as e:
#         st.error(traceback.format_exc())
#         return f"Error formatting prediction results: {str(e)}"

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

# Initialize knowledge base and disease model
kb_data = None
disease_model = None
with st.spinner("Loading resources..."):
    try:
        kb_data = load_knowledge_base()
        if kb_data:
            st.success("Knowledge base loaded successfully!")
        
        # Attempt to load disease model (but don't block app if it fails)
        try:
            disease_model = load_disease_model()
            if disease_model:
                st.success("Disease prediction model loaded successfully!")
        except Exception as e:
            st.warning("Disease prediction model could not be loaded. Some features may be unavailable.")
            st.error(f"Error: {str(e)}")
    except Exception as e:
        st.error(f"Failed to initialize resources: {str(e)}")

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

# Function to analyze image using Gemini Pro Vision
def analyze_agricultural_image(image, prompt_text=None):
    try:
        # Convert the image to the format expected by Gemini
        if hasattr(image, 'convert'):
            # If it's a PIL Image
            img_bytes_io = BytesIO()
            image.save(img_bytes_io, format='JPEG')
            img_bytes = img_bytes_io.getvalue()
        else:
            # If it's already bytes
            img_bytes = image
        
        # Create multimodal prompt
        if not prompt_text:
            prompt_text = """Analyze this agricultural image and provide:
            1. Plant/crop identification
            2. Growth stage assessment
            3. Health status (identify any visible diseases, pests, nutrient deficiencies)
            4. Recommended actions for the farmer
            
            If the image is unclear or you can't identify the plant/crop with certainty, 
            please indicate this and provide possible options.
            """
        
        # Initialize Gemini Vision model with correct model name
        # Try gemini-pro-vision first (older model name)
        try:
            model = genai.GenerativeModel('gemini-pro-vision')
            # Generate content with image and prompt
            response = model.generate_content([prompt_text, {"mime_type": "image/jpeg", "data": img_bytes}])
            return response.text
        except Exception as e1:
            # If that fails, try gemini-1.5-pro (newer model might handle images directly)
            try:
                model = genai.GenerativeModel('gemini-1.5-pro')
                # Generate content with image and prompt
                response = model.generate_content([prompt_text, {"mime_type": "image/jpeg", "data": img_bytes}])
                return response.text
            except Exception as e2:
                return f"Error analyzing image: Failed with both model attempts. First error: {str(e1)}. Second error: {str(e2)}"
    
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

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
    st.session_state.current_tab = "text"

# Function to set current tab
def set_tab(tab_name):
    st.session_state.current_tab = tab_name

# Add toggle for text-to-speech responses
tts_enabled = st.sidebar.toggle("Enable Voice Responses", value=True)

# Create tabs for different input methods
tab_col1, tab_col2, tab_col3 = st.columns([1, 1, 1])
with tab_col1:
    if st.button("📝 Text/Voice Input", use_container_width=True, 
                 type="primary" if st.session_state.current_tab == "text" else "secondary"):
        set_tab("text")
with tab_col2:
    if st.button("🖼️ Image Analysis", use_container_width=True,
                type="primary" if st.session_state.current_tab == "image" else "secondary"):
        set_tab("image")
# with tab_col3:
#     if st.button("🔬 Disease Prediction", use_container_width=True,
#                 type="primary" if st.session_state.current_tab == "disease_predict" else "secondary"):
#         set_tab("disease_predict")

# Text/Voice tab content
if st.session_state.current_tab == "text":
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

# Image Analysis tab content
elif st.session_state.current_tab == "image":
    st.subheader("Plant/Crop Image Analysis")
    
    # Image upload
    uploaded_file = st.file_uploader("Upload an image of a plant, crop, or agricultural issue", 
                                     type=["jpg", "jpeg", "png"],
                                     key="image_analysis_uploader")
    
    # Optional custom prompt
    custom_prompt = st.text_area("Optional: Add specific questions about this image", 
                                height=100, 
                                help="Leave blank for general analysis, or ask specific questions like 'What disease does this tomato plant have?' or 'Is this crop ready for harvest?'")
    
    analyze_button = st.button("🔍 Analyze Image")
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Image analysis
        if analyze_button:
            with st.spinner("Analyzing image..."):
                # Prepare the prompt
                if custom_prompt:
                    prompt = f"""I'm an agricultural expert analyzing this image. 
                    
                    User's specific question: {custom_prompt}
                    
                    Please provide a detailed analysis addressing the user's question, along with:
                    1. Plant/crop identification
                    2. Growth stage assessment
                    3. Health status (identify any visible diseases, pests, nutrient deficiencies)
                    4. Recommended actions for the farmer
                    """
                else:
                    prompt = None  # Use default prompt in the analyze_agricultural_image function
                
                # Get image analysis
                analysis_result = analyze_agricultural_image(image, prompt)
                
                # Store in chat history
                st.session_state.chat_history.append(f"**User:** *Uploaded an image for analysis*")
                st.session_state.chat_history.append(f"**AI:** {analysis_result}")
                
                # Display results
                st.subheader("📊 Image Analysis Results:")
                st.markdown(analysis_result)
                
                # Convert response to speech if enabled
                if tts_enabled:
                    audio_html = text_to_speech(analysis_result)
                    if audio_html:
                        st.markdown(audio_html, unsafe_allow_html=True)
    else:
        st.info("Please upload an image to analyze")

# Disease Prediction tab content
elif st.session_state.current_tab == "disease_predict":
    st.subheader("🌿 Plant Disease Prediction")
    
    # Image upload for disease prediction
    uploaded_file = st.file_uploader("Upload an image of a potentially diseased plant", 
                                     type=["jpg", "jpeg", "png"],
                                     key="disease_prediction_uploader")
    
    # Prediction button
    predict_button = st.button("🔍 Predict Disease")
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Plant Image", use_container_width=True)
        
        # Disease prediction
        if predict_button:
            if disease_model is None:
                st.error("Disease prediction model is not loaded. Please check model configuration.")
            else:
                with st.spinner("Predicting disease..."):
                    # Make prediction
                    prediction_result = predict_disease(image)
                    
                    # Store in chat history
                    st.session_state.chat_history.append(f"**User:** *Uploaded an image for disease prediction*")
                    st.session_state.chat_history.append(f"**AI:** {prediction_result}")
                    
                    # Display results
                    st.subheader("🔬 Disease Prediction Results:")
                    st.markdown(prediction_result)
                    
                    # Convert response to speech if enabled
                    if tts_enabled:
                        # Extract main points for speech (to keep it concise)
                        speech_text = "I've analyzed your plant image. " + prediction_result.split("###")[1].strip()
                        audio_html = text_to_speech(speech_text)
                        if audio_html:
                            st.markdown(audio_html, unsafe_allow_html=True)
                            
                    # Option to get more detailed information from Gemini
                    if st.button("🔍 Get Additional Information from AI"):
                        with st.spinner("Getting detailed information..."):
                            # Try to extract the predicted disease from the result
                            try:
                                if "Prediction:" in prediction_result:
                                    disease_name = prediction_result.split("Prediction:")[1].split("\n")[0].strip()
                                else:
                                    disease_name = "the detected plant condition"
                            except:
                                disease_name = "the detected plant condition"
                                
                            # Get detailed information from Gemini
                            question = f"Provide detailed information about {disease_name} in plants, including symptoms, causes, prevention, and treatment options."
                            detailed_info = get_rag_response(question)
                            
                            # Display the information
                            st.subheader(f"📚 Detailed Information")
                            st.markdown(detailed_info)
    else:
        st.info("Please upload an image to predict plant disease")

# Display chat history
st.subheader("📝 Chat History")
for chat in st.session_state.chat_history:
    st.markdown(chat)

# Display PDF info
st.sidebar.header("About")
st.sidebar.info("This chatbot answers questions based on agricultural knowledge from 'data.pdf', analyzes plant/crop images, and predicts plant diseases using a pretrained model.")

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

# Add image analysis instructions
st.sidebar.header("Image Analysis")
st.sidebar.info("""
- Navigate to the 'Image Analysis' tab
- Upload an image of your plant, crop, or agricultural issue
- Add specific questions about the image (optional)
- Click 'Analyze Image' to get AI insights
""")

# Add disease prediction instructions
st.sidebar.header("Disease Prediction")
st.sidebar.info("""
- Navigate to the 'Disease Prediction' tab
- Upload an image of a potentially diseased plant
- Click 'Predict Disease' to use the pretrained model
- The model will identify the disease and provide recommendations
""")

# Requirements instruction
st.sidebar.header("Setup Requirements")
st.sidebar.code("""
pip install streamlit google-generativeai langchain langchain-community PyMuPDF scikit-learn python-dotenv SpeechRecognition gTTS Pillow tensorflow
""")