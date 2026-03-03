import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

# ---------------------------
# Load Environment Variables
# ---------------------------
load_dotenv()

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="AI Calculator Agent", page_icon="🤖")
st.title("🤖 AI Calculator Agent")
st.write("Ask any mathematical question!")

# ---------------------------
# Calculator Function
# ---------------------------


def calculator(expression: str):
    try:
        expression = expression.replace(" ", "")
        result = eval(expression)
        return str(result)
    except Exception:
        return "Invalid mathematical expression."

# ---------------------------
# Convert Function into Tool
# ---------------------------


calc_tool = Tool(
    name="Calculator",
    func=calculator,
    description=(
        "Use this tool to solve mathematical expressions "
        "like 45*89+120 or 100/5."
    )
)

# ---------------------------
# Initialize Groq LLM
# ---------------------------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# ---------------------------
# Create Agent
# ---------------------------
agent = initialize_agent(
    tools=[calc_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ---------------------------
# User Input Section
# ---------------------------
user_input = st.text_input("Enter your question:")

if st.button("Calculate"):
    if user_input:
        with st.spinner("Thinking..."):
            try:
                response = agent.run(user_input)
                st.success(f"Answer: {response}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a question.")
