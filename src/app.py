import streamlit as st
import sounddevice as sd
import scipy.io.wavfile as wav
import whisper
import os
import numpy as np
import asyncio
import edge_tts
import base64
import re
from agent import generate_ai_diagnostic, get_cnn_score

st.set_page_config(page_title="Interactive AI Therapist", page_icon="🤖")

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

model = load_whisper()

async def generate_audio(text, filename="tts.mp3"):
    communicate = edge_tts.Communicate(text, "en-US-AnaNeural")
    await communicate.save(filename)

def play_audio(filename):
    with open(filename, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

if 'word_list' not in st.session_state:
    st.session_state.word_list = ["APPLE", "ELEPHANT", "CHOCOLATE", "STRAWBERRY"]
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'step' not in st.session_state:
    st.session_state.step = "intro"

st.title("🤖 Interactive AI Speech Therapist")

if st.session_state.current_index < len(st.session_state.word_list):
    target_word = st.session_state.word_list[st.session_state.current_index]
    
    st.markdown(f"### 🎯 Let's practice: **:blue[{target_word}]**")

    if st.session_state.step == "intro":
        intro_text = f"Hi! Let's practice. Can you say the word... {target_word}?"
        st.info(f"🤖 AI: {intro_text}")
        
        with st.spinner("Preparing audio..."):
            asyncio.run(generate_audio(intro_text, "intro.mp3"))
        play_audio("intro.mp3")
        
        if st.button("🎙️ I'm ready! Start Recording (3s)"):
            st.session_state.step = "recording"
            st.rerun()

    elif st.session_state.step == "recording":
        st.warning("🎤 Recording started! Please speak now...")
        
        fs = 22050
        duration = 3
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
        sd.wait()
        
        if not os.path.exists("dataset"):
            os.makedirs("dataset")
        audio_path = "dataset/input.wav"
        wav.write(audio_path, fs, recording)
        st.success("✅ Audio captured!")
        
        with st.spinner("🎧 AI is listening..."):
            result = model.transcribe(audio_path, language="en")
            st.session_state.spoken_text = result["text"].strip().upper()
            
            cleaned_text = re.sub(r'[^A-Z]', '', st.session_state.spoken_text)
            
        if not cleaned_text:
            st.session_state.cnn_score = 0
            st.session_state.feedback = "I didn't hear anything! 🙊 Could you try speaking a little louder?"
        else:
            with st.spinner("🧠 AI CNN is scoring your acoustic waves..."):
                cnn_score = get_cnn_score(audio_path)
                st.session_state.cnn_score = cnn_score
                
            with st.spinner("👨‍⚕️ AI is diagnosing your pronunciation..."):
                if target_word not in st.session_state.spoken_text and cnn_score < 30:
                    st.session_state.feedback = f"Wait a minute! 🤔 I heard '{st.session_state.spoken_text}' instead of '{target_word}'. Let's focus and try saying the right word!"
                elif cnn_score >= 85:
                    st.session_state.feedback = f"Perfect! You got a great score of {cnn_score}%. Your pronunciation was spot on!"
                else:
                    st.session_state.feedback = generate_ai_diagnostic(target_word, st.session_state.spoken_text, cnn_score)
        
        st.session_state.step = "feedback"
        st.rerun()

    elif st.session_state.step == "feedback":
        st.write(f"**What the AI heard:** _{st.session_state.spoken_text if st.session_state.spoken_text else '[Silence]'}_")
        
        st.subheader("📊 Pronunciation Score (by CNN Model)")
        st.progress(st.session_state.cnn_score / 100.0)
        st.metric(label="Score", value=f"{st.session_state.cnn_score}%")
        
        st.info(f"💡 AI Diagnostic:\n{st.session_state.feedback}")
        
        with st.spinner("Generating AI voice..."):
            asyncio.run(generate_audio(st.session_state.feedback, "feedback.mp3"))
        play_audio("feedback.mp3")
        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➡️ Next Word"):
                st.session_state.current_index += 1
                st.session_state.step = "intro"
                st.rerun()
        with col2:
            if st.button("🔄 Try Again"):
                st.session_state.step = "intro"
                st.rerun()

else:
    st.balloons()
    st.success("🎉 Wow! You completed all the words today! Great job!")
    if st.button("🔄 Restart Session"):
        st.session_state.current_index = 0
        st.session_state.step = "intro"
        st.rerun()