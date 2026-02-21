import os
import numpy as np
import librosa
import tensorflow as tf
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

print("🧠 جاري تحميل الموديل الذكي (CNN)...")
try:
    cnn_model = tf.keras.models.load_model("dataset/speech_cnn_model.h5", compile=False)
    print("✅ تم تحميل الموديل بنجاح!")
except Exception as e:
    print(f"⚠️ تحذير: لم يتم العثور على الموديل: {e}")
    cnn_model = None

def extract_mfcc(file_path, n_mfcc=40):
    try:
        audio, sample_rate = librosa.load(file_path, sr=16000, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        return None

def get_cnn_score(audio_path="dataset/input.wav"):
    if cnn_model is None:
        return 50
        
    features = extract_mfcc(audio_path)
    if features is None:
        return 50
        
    features = features.reshape(1, 40, 1)
    prediction = cnn_model.predict(features, verbose=0)
    score = min(max(int(prediction[0][0]), 0), 100)
    
    return score

def generate_ai_diagnostic(target_word, spoken_text, cnn_score, age=8):
    prompt = f"""
    You are an AI Speech Therapist for children (around {age} years old).
    
    - The child tried to say the word: "{target_word}"
    - Our STT model heard: "{spoken_text}"
    - Our Acoustic CNN Model analyzed the sound waves and gave a quality score of: {cnn_score}/100.

    Diagnostic Logic:
    1. If the CNN score is below 80 but the STT heard the correct word, it means the pronunciation was correct but muffled, mumbled, or lacked a good accent.
    2. If the STT heard a different word, identify which phonetic sound (letter) they mispronounced based on the difference.
    
    Task:
    Provide a gentle 3-sentence feedback. 
    First, mention their score of {cnn_score}%. 
    Second, diagnose their exact pronunciation issue based on the logic above. 
    Third, give physical advice (how to position tongue/lips/teeth) to fix it.
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful and playful speech therapist."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=150
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Oops! I had a little trouble thinking. Let's try saying '{target_word}' again!"