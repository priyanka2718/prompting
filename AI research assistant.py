import streamlit as st
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Streamlit UI
st.title("🔎 AI Research Assistant")
st.write("Ask any research question")

# User input
query = st.text_input("Enter your question")

# Load LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# Wikipedia Tool
wiki = WikipediaAPIWrapper()

tools = [
    Tool(
        name="Wikipedia",
        func=wiki.run,
        description="Use this tool to search information from Wikipedia"
    )
]

# Create agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run agent
if st.button("Search"):
    if query:
        response = agent.run(query)
        st.write("### Answer")
        st.write(response)
