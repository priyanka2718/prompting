import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()

st.title("🤖 AI Python Code Explainer")

code_input = st.text_area("Paste your Python code here")

# Load LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

if st.button("Explain Code"):

    prompt = f"""
You are a Python teacher.

Explain the following Python code step-by-step.

STRICT RULES:
- Do NOT give paragraph explanation
- Explain each line separately
- Use ONLY this format

Step 1 - explanation
Step 2 - explanation
Step 3 - explanation
Step 4 - explanation
Step 5 - explanation

Python Code:
{code_input}
"""

    response = llm.invoke(prompt)

    result = response.content

    st.write("### Explanation")

    lines = result.split("\n")

    for line in lines:
        if line.strip():
            st.write(line)
