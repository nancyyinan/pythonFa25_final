import streamlit as st
from transformers import pipeline
from huggingface_hub import InferenceClient
from PIL import Image
import io

# ================= 1. Configuration =================

# Setup Page Config (Optional, makes it look nicer)
st.set_page_config(page_title="AI Mood-to-Image", page_icon="🎨")

# 🔐 SECURITY: Retrieve Token from Streamlit Secrets
# This prevents the token from being exposed and invalidated by GitHub
try:
    HF_API_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    st.error("⚠️ Token not found! Please set 'HF_TOKEN' in Streamlit Cloud Secrets.")
    st.stop()

# Model ID for Image Generation
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# ================= 2. Model Loading & Functions =================

# A. Load Sentiment Analysis Model
# Caching prevents reloading the model on every interaction
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# B. Load Image Captioning Model
@st.cache_resource
def load_caption_pipeline():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# C. Generate Image Function (Using Official InferenceClient)
def generate_image(prompt_text):
    """
    Generates an image based on the text prompt using Hugging Face Inference API.
    """
    client = InferenceClient(token=HF_API_TOKEN)
    try:
        # Directly returns a PIL Image object
        image = client.text_to_image(prompt_text, model=MODEL_ID)
        return image
    except Exception as e:
        return f"Error: {e}"

# ================= 3. UI Layout (Streamlit) =================

st.title("🎨 AI Mood-to-Image Artist")
st.markdown("### Social Media Sentiment Analysis Final Project")
st.info("Workflow: Input Text/Image -> AI Analyzes Sentiment -> AI Generates Abstract Art")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Project Settings")
    st.write("**Models Used:**")
    st.caption("1. Sentiment: `cardiffnlp/twitter-roberta-base`")
    st.caption("2. Generation: `stabilityai/sdxl-base-1.0`")
    st.caption("3. Captioning: `Salesforce/blip-captioning`")
    
    # Token Status Check (Hidden partial token for safety)
    if HF_API_TOKEN:
        st.success("✅ HF Token Loaded")

# --- Input Method Selection ---
input_method = st.radio("Choose Input Method:", ["📝 Text Input", "🖼️ Upload Image"])

final_text_input = ""

if input_method == "📝 Text Input":
    final_text_input = st.text_area("Enter Tweet/Text:", "I am so happy that I finished my final project!")

elif input_method == "🖼️ Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        
        # Perform Image Captioning
        with st.spinner('AI is analyzing the image content...'):
            caption_pipe = load_caption_pipeline()
            # Handle potential list or dict return
            caption_output = caption_pipe(image)
            caption_result = caption_output[0]['generated_text']
            st.success(f"AI Detected Content: '{caption_result}'")
            final_text_input = caption_result

# --- Main Processing Logic ---
if st.button("Analyze & Generate"):
    if not final_text_input:
        st.warning("Please input text or upload an image first.")
    else:
        # 1. Sentiment Analysis
        st.divider()
        st.subheader("1. Sentiment Analysis")
        
        with st.spinner('Analyzing sentiment...'):
            sentiment_pipe = load_sentiment_pipeline()
            # Truncate text to 512 chars to fit model limits
            result = sentiment_pipe(final_text_input[:512])[0]
            
            label = result['label']  # positive / negative / neutral
            score = result['score']
            
            # Display Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Detected Sentiment", label.upper(), f"{score:.2f}")
            with col2:
                # Dynamic Emoji
                if "positive" in label:
                    st.markdown("# 😄")
                elif "negative" in label:
                    st.markdown("# 😢")
                else:
                    st.markdown("# 😐")

        # 2. Generative Art
        st.divider()
        st.subheader("2. Generative Art Visualization")
        
        # Dynamic Prompting based on Sentiment
        if "positive" in label:
            style = "vibrant colors, sunshine, joyful, masterpiece, digital art, 8k"
        elif "negative" in label:
            style = "dark gloomy colors, rain, abstract, sad atmosphere, charcoal style"
        else:
            style = "minimalist, pastel colors, calm, balanced, geometric shapes"
            
        prompt = f"abstract art representing '{final_text_input}', {style}"
        st.caption(f"🎨 **Generation Prompt:** {prompt}")

        # Call Image Generation API
        with st.spinner('AI is painting... (This may take a moment for model cold-start)'):
            generated_result = generate_image(prompt)
            
            # Check if result is an image or an error string
            if isinstance(generated_result, Image.Image):
                st.image(generated_result, caption=f"Visualized Sentiment: {label.upper()}", use_column_width=True)
            else:
                # Handle Errors
                st.error("Image Generation Failed.")
                st.error(f"Error Details: {generated_result}")
                st.info("Note: If the error mentions '503' or 'loading', please wait 30 seconds and try again.")

# ================= 4. Dataset Info (For Assignment) =================
st.divider()
with st.expander("View Dataset Loading Code (For Assignment Reference)"):
    st.code("""
from datasets import load_dataset
# Load the tweet_eval sentiment subset
ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    """, language='python')
