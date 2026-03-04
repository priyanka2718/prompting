from gtts import gTTS
import streamlit as st

st.title("AI Text to Speech Generator")

text = st.text_area("Enter text")

if st.button("Generate Voice"):
    tts = gTTS(text=text, lang='en')
    tts.save("voice.mp3")

    audio_file = open("voice.mp3", "rb")
    audio_bytes = audio_file.read()

    st.audio(audio_bytes, format="audio/mp3")explain this code line by line
