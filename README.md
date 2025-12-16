# 🎨 AI Mood-to-Image Artist (Social Media Sentiment Analysis)

## Project Overview
This project is a web-based application built with **Streamlit** that explores the intersection of Natural Language Processing (NLP) and Generative Art. The application analyzes the sentiment of a user's input (text or image) and automatically generates an abstract artistic representation of that emotion using the Stable Diffusion model.

**App Link:** [Click here to view the deployed app](https://pythonfa25final-ulqosjgq8kjttb7nt54885.streamlit.app/) 

---

## 🤖 Models Used
This application integrates three distinct state-of-the-art AI models hosted on Hugging Face:

1.  **Sentiment Analysis:** `cardiffnlp/twitter-roberta-base-sentiment-latest`
    * **Description:** A RoBERTa-base model trained on ~58 million tweets and fine-tuned for sentiment analysis.
    * **Role:** It classifies the user's input text into "Positive", "Negative", or "Neutral" with a confidence score.
    
2.  **Image Generation:** `stabilityai/stable-diffusion-xl-base-1.0` (SDXL)
    * **Description:** A powerful latent text-to-image diffusion model capable of generating photorealistic and artistic images.
    * **Role:** It takes the sentiment and a dynamically constructed prompt to generate the final abstract artwork.

3.  **Image Captioning:** `Salesforce/blip-image-captioning-base`
    * **Description:** The BLIP (Bootstrapping Language-Image Pre-training) model capable of generating captions for visual content.
    * **Role:** If the user uploads an image instead of text, this model converts the visual content into a text description, which is then fed into the sentiment analyzer.

---

## 📊 Data Background
While this application primarily performs inference using pre-trained models, the core sentiment analysis capabilities are rooted in the **TweetEval** benchmark dataset.

* **Dataset:** `cardiffnlp/tweet_eval`
* **Context:** This dataset consists of seven heterogeneous tasks in Twitter sentiment analysis (e.g., irony, hate speech, sentiment).
* **Relevance:** The underlying model (RoBERTa) was fine-tuned on the `sentiment` subset of this data, making it highly effective at understanding the nuances, slang, and brevity typical of social media text.

*(Note: Code for loading this dataset is included in the application for assignment reference.)*

---

## 🛠️ Development Process
The development of this application followed an iterative pipeline approach:

1.  **Environment Setup:** Created a virtual environment and installed necessary libraries (`streamlit`, `transformers`, `huggingface_hub`, `torch`).
2.  **Backend Integration:**
    * Utilized Hugging Face's `pipeline` for efficient local inference of the sentiment and captioning models.
    * Implemented the `InferenceClient` API for the heavier Stable Diffusion model to offload computation to the cloud and ensure fast generation times.
3.  **Frontend Interface:** Built a user-friendly interface using Streamlit, featuring a sidebar for model details, dynamic emoji feedback based on sentiment scores, and error handling for API connectivity.
4.  **Logic & Prompt Engineering:** Designed a conditional logic system that alters the artistic style prompts (e.g., "vibrant colors" for positive, "gloomy charcoal" for negative) based on the detected sentiment label.
5.  **Deployment & Security:** Deployed the app to Streamlit Cloud, ensuring sensitive API keys were managed securely via `st.secrets` rather than hardcoded in the repository.

---

## 🚀 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/nancyyinan/pythonFa25_final.git](https://github.com/nancyyinan/pythonFa25_final.git)
    cd pythonFa25_final
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up API Keys:**
    * Create a `.streamlit/secrets.toml` file in the root directory.
    * Add your Hugging Face token:
        ```toml
        HF_TOKEN = "your_huggingface_token_here"
        ```

4.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

---
*Final Project for Python Class - Fall 2025*
