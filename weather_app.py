import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# -------- LOAD ENV --------
load_dotenv()

# -------- STREAMLIT CONFIG --------
st.set_page_config(page_title="Weather Agent", page_icon="🌦️")
st.title("🌦️ AI Weather Assistant")
st.write("Ask weather-related questions like:")
st.write("• Should I carry umbrella in Chennai?")
st.write("• I am going to Mumbai today")
st.write("• Weather in Delhi")

# -------- GROQ LLM --------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# -------- WEATHER TOOL --------
def get_weather(city):
    api_key = os.getenv("WEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    response = requests.get(url).json()

    if response.get("cod") != 200:
        return None

    temp = response["main"]["temp"]
    weather = response["weather"][0]["description"]

    return {
        "temperature": temp,
        "condition": weather
    }

# -------- AGENT FUNCTION --------
def weather_agent(question):

    # -------- STEP 1: Extract City Using LLM --------
    extract_prompt = f"""
Extract only the city name from this sentence.
Return only the city name. Do not add extra words.

Sentence:
{question}
"""
    city_response = llm.invoke(extract_prompt)
    city = city_response.content.strip()

    # -------- STEP 2: Call Weather Tool --------
    weather_data = get_weather(city)

    if weather_data is None:
        return "City not found. Please enter a valid city name.", "City not found."

    condition = weather_data["condition"]
    temp = weather_data["temperature"]

    weather_info = f"Temperature: {temp}°C, Condition: {condition}"

    # -------- STEP 3: Umbrella Decision Logic --------
    prompt = f"""
You are a helpful weather assistant.

Weather condition: {condition}
Temperature: {temp}°C

If weather condition contains:
- rain
- drizzle
- thunderstorm
- shower

Recommend umbrella.

Otherwise, say umbrella not required.

Give clear and confident answer mentioning the city: {city}.
"""

    response = llm.invoke(prompt)

    return response.content, weather_info


# -------- UI --------
question = st.text_input("Ask your weather question:")

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Checking weather..."):
            answer, weather_info = weather_agent(question)

        st.subheader("🌡 Weather Data")
        st.write(weather_info)

        st.subheader("🤖 AI Recommendation")
        st.write(answer)
