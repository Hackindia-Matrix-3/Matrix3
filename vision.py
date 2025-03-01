import streamlit as st
import os
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()

# Retrieve API key from environment variables
api_key = os.getenv("GOOGLE_API_KEY")

# Debugging: Print API Key (Remove this in production)
if not api_key:
    st.error("GOOGLE_API_KEY not found. Please set it in the .env file.")
    raise ValueError("GOOGLE_API_KEY not found.")

# Configure Gemini API
genai.configure(api_key=api_key)

# Function to get Gemini response
def get_gemini_response(input_text, image=None):
    model = genai.GenerativeModel('gemini-pro-vision')
    if input_text and image:
        response = model.generate_content([input_text, image])
    elif input_text:
        response = model.generate_content(input_text)
    else:
        response = model.generate_content(image)
    
    return response.text

# Initialize Streamlit app
st.set_page_config(page_title="Gemini Image Q&A")
st.header("Gemini Application")

# Input text prompt
input_text = st.text_input("Enter your question:", key="input")

# Upload image
uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Submit button
if st.button("Tell me about the image"):
    try:
        response = get_gemini_response(input_text, image)
        st.subheader("Response:")
        st.write(response)
    except Exception as e:
        st.error(f"Error: {e}")