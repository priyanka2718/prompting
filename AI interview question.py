import streamlit as st
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

st.title("🎯 AI Interview Q&A Generator")

topic = st.text_input("Enter interview topic (Example: Python Developer)")

# Load LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# Tool function


def generate_qa(topic):

    prompt = f"""
You are a technical interviewer.

Generate 5 interview questions and answers about the topic below.

Rules:
- Use simple language
- Format clearly
- Provide both question and answer

Topic:
{topic}

Format:

Question 1:
Answer:

Question 2:
Answer:

Question 3:
Answer:

Question 4:
Answer:

Question 5:
Answer:
"""

    response = llm.invoke(prompt)

    return response.content


# Tool definition
tools = [
    Tool(
        name="Interview QA Tool",
        func=generate_qa,
        description="Generates interview questions and answers for a topic"
    )
]

# Create agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Button
if st.button("Generate Interview Q&A"):

    if topic:

        result = agent.run(topic)

        st.write("### Interview Questions and Answers")

        lines = result.split("\n")

        for line in lines:
            if line.strip():
                st.write(line)
