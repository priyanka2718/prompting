import streamlit as st
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

st.title("📝 AI Grammar Corrector")

text_input = st.text_area("Enter a sentence with grammar mistakes")

# Load LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# Tool function


def grammar_corrector(text):

    prompt = f"""
You are an English grammar expert.

Correct the grammar of the following sentence.

Rules:
- Return only the corrected sentence
- Do not give explanations

Sentence:
{text}
"""

    response = llm.invoke(prompt)

    return response.content


# Define Tool
tools = [
    Tool(
        name="Grammar Corrector Tool",
        func=grammar_corrector,
        description="Corrects grammar mistakes in sentences"
    )
]

# Create Agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Button
if st.button("Correct Grammar"):

    if text_input:

        result = agent.run(text_input)

        st.write("### Corrected Sentence")

        st.success(result)
