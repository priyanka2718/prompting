import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()

st.title("📚 AI Study Planner")
st.write("Generate a structured study plan")

question = st.text_input("Ask your study question")

# Fast Groq model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

if st.button("Generate Plan"):

    prompt = f"""
You are an AI Study Planner.

User Question:
{question}

Task:
Create a structured learning roadmap.

Rules:
- Do NOT write paragraphs
- Output ONLY step-by-step study plan
- Use the exact format below

Format:

Subject: <subject name>

Total Days Required: <number>

Study Plan:

Day 1 - <topic>
Day 2 - <topic>
Day 3 - <topic>
Day 4 - <topic>
Day 5 - <topic>

Continue until the learning plan is complete.
"""

    response = llm.invoke(prompt)

    result = response.content

    # Show output line by line
    lines = result.split("\n")

    st.write("### Study Plan")

    for line in lines:
        if line.strip():
            st.write(line)
