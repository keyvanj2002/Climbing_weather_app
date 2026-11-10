"""Streamlit frontend to help climbers understand best weather windows."""
import asyncio
import os
import streamlit as st
for key in [
    "GROQ_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGSMITH_TRACING",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
]:
    if key in st.secrets:
        os.environ[key] = str(st.secrets[key])

import streamlit as st
from dotenv import load_dotenv

from weather_agent import create_weather_agent

st.set_page_config(page_title="climbing Weather Assistant", page_icon="🌤️", layout="centered")
st.title("🌤️ climbing Weather Assistant")
st.caption("daily and hourly weather forecasts for climbers • Powered by LangChain and Groq")

load_dotenv()

def api_key():
    # Streamlit Cloud: Settings → Secrets. Local: .env file.
    try:
        return st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    except (FileNotFoundError, KeyError):
        return os.getenv("GROQ_API_KEY")

with st.sidebar:
    st.subheader("About")
    st.write("Ask about climbing forecasts, wind, rain, UV, or other conditions.")
    st.info("Forecasts are not live observations. Hourly coverage can be shorter than daily coverage.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("e.g. I want to climb in Grampains. How windy will Grampians be tomorrow afternoon?")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        key = api_key()
        if not key:
            st.error("Missing GROQ_API_KEY. Set it in Streamlit secrets (cloud) or .env (local).")
        else:
            with st.spinner("Checking the forecast..."):
                try:
                    agent = create_weather_agent(key)
                    # Send a bounded amount of chat history to control token costs.
                    recent = st.session_state.messages[-12:]
                    result = asyncio.run(agent.ainvoke({
                        "messages": [(m["role"], m["content"]) for m in recent],
                    }))
                    content = result["messages"][-1].content
                    answer = content if isinstance(content, str) else str(content)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as exc:
                    st.error("Could not retrieve the forecast. Please try again.")
                    # Do not expose detailed errors or credentials in a public app.
                    print(f"Weather agent error: {type(exc).__name__}: {exc}")
