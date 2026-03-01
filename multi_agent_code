import subprocess
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# -------- LOAD ENV --------
load_dotenv()

# -------- STREAMLIT CONFIG --------
st.set_page_config(page_title="AI Coding Assistant", page_icon="🤖")
st.title("🤖 Multi-Agent AI Coding Assistant")
st.write("Bug Detector Agent + Code Optimizer Agent")

# -------- GROQ LLM --------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# -------- TOOL: Python Execution --------
def run_python_code(code):
    try:
        with open("temp.py", "w") as f:
            f.write(code)

        result = subprocess.run(
            ["python", "temp.py"],
            capture_output=True,
            text=True
        )

        if result.stderr:
            return False, result.stderr
        return True, result.stdout

    except Exception as e:
        return False, str(e)

# -------- AGENT 1 --------
def bug_detector(code, execution_result):
    prompt = f"""
You are a Python bug detection expert.

Here is the code:
{code}

Execution result:
{execution_result}

Explain the error clearly. If no error, say code runs correctly.
"""
    response = llm.invoke(prompt)
    return response.content

# -------- AGENT 2 --------
def optimizer(code):
    prompt = f"""
You are a senior Python developer.

Improve this code and optimize it for readability and best practices:

{code}
"""
    response = llm.invoke(prompt)
    return response.content


# -------- UI INPUT --------
user_code = st.text_area("Paste your Python code here:", height=250)

if st.button("Analyze Code"):
    if user_code.strip() == "":
        st.warning("Please enter some code.")
    else:
        with st.spinner("Analyzing..."):
            success, execution_output = run_python_code(user_code)

            bug_result = bug_detector(user_code, execution_output)
            opt_result = optimizer(user_code)

        # -------- TABS --------
        tab1, tab2, tab3 = st.tabs(
            ["🐞 Bug Detector", "⚡ Optimizer", "▶ Execution Output"]
        )

        with tab1:
            st.subheader("Bug Detector Agent Output")
            st.write(bug_result)

        with tab2:
            st.subheader("Optimizer Agent Output")
            st.write(opt_result)

        with tab3:
            st.subheader("Execution Output")
            if success:
                st.success("Code executed successfully ✅")
                st.code(execution_output)
            else:
                st.error("Execution Error ❌")
                st.code(execution_output)
