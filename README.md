# 🤖 SmartSpeech V2: Interactive AI Speech Therapist

An AI-powered speech therapist designed to help children improve their English pronunciation using a Hybrid AI architecture. This project was built as a Proof of Concept (PoC) for a final year engineering project (PFE).

## 🚀 System Architecture
This project uses a Hybrid AI Pipeline combining acoustic analysis and Generative AI:
1. **Whisper (Speech-to-Text):** Transcribes the spoken audio into text.
2. **Custom CNN Model (Acoustic Analyzer):** A Convolutional Neural Network trained on the `speechocean762` dataset. It extracts **MFCC** (Mel-frequency cepstral coefficients) from the raw audio waves and outputs a purely acoustic quality score (0-100).
3. **Groq / Llama 3 (Diagnostic Brain):** An LLM that compares the CNN's acoustic score with the STT output to generate physical, actionable advice (e.g., tongue/lip placement) for the child.
4. **Edge-TTS:** Provides interactive, child-friendly voice feedback.

## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow, Keras, Librosa
- **AI Models:** Whisper (Local), Llama-3.1-8b (via Groq API)
- **Frontend:** Streamlit
- **Environment:** Docker & Docker Compose

## ⚙️ How to Run Locally
1. Clone the repository.
2. Create a `.env` file in the root directory and add your Groq API key: `GROQ_API_KEY="your_key_here"`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the app: `streamlit run src/app.py`
