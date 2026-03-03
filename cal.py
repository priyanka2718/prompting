import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ---------------------------
# Load Environment Variables
# ---------------------------
load_dotenv()

# Get API Key
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()

# ---------------------------
# Initialize Groq Model
# ---------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI Calculator", page_icon="🧮")
st.title("🧮 Simple AI Calculator")
st.write("Enter a math expression like 2+3, 3*2, 10/5")

# ---------------------------
# Safe Math Evaluator
# ---------------------------
def calculate(expression: str):
    try:
        if not re.match(r"^[0-9+\-*/(). ]+$", expression):
            return "Invalid input"

        result = eval(expression)
        return str(result)

    except Exception:
        return "Invalid mathematical expression"

# ---------------------------
# User Input
# ---------------------------
user_input = st.text_input("Enter expression:")

if st.button("Calculate"):
    if user_input:
        with st.spinner("Calculating..."):
            answer = calculate(user_input)
            st.success(f"Answer: {answer}")
    else:
        st.warning("Please enter a valid math expression.")
