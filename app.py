#import os
#import getpass

#if not  os.getenv("OPENAI_API_KEY"):
  #  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key (https://platform.openai.com/account/api-keys):\n")

#from langchain_openai import OpenAI

#chat = OpenAI(model="gpt-4o-mini", temperature=0.5)

#prompt = "Explain what is AI Agentics in simple terms.  Make it fun and easy to understand. Use examples."

#respoonse = chat.invoke(prompt)

#print(respoonse)
# app.py
import streamlit as st
import os
import asyncio
import datetime
import time
from typing import Any
from langchain_openai import OpenAI

# ------------------------------
# Helpers
# ------------------------------
def extract_text(response: Any) -> str:
    """Extract the main text from common response shapes."""
    try:
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, (list, tuple)) and len(content) > 0:
                first = content[0]
                if isinstance(first, dict) and "text" in first:
                    return first["text"]
                if hasattr(first, "text"):
                    return first.text
        if isinstance(response, dict):
            c = response.get("content")
            if isinstance(c, (list, tuple)) and len(c) > 0:
                first = c[0]
                if isinstance(first, dict) and "text" in first:
                    return first["text"]
        return str(response)
    except Exception:
        return str(response)

def safe_rerun():
    """
    Try to call st.rerun() (preferred). If not available, force rerun by updating query params.
    Uses st.query_params assignment (non-experimental).
    """
    try:
        st.rerun()
    except Exception:
        # Fallback: change query params to force a rerun
        try:
            st.query_params = {"_rerun": str(time.time())}
        except Exception:
            # As a last resort, do nothing (can't force rerun)
            pass

# ------------------------------
# Page config + CSS
# ------------------------------
st.set_page_config(page_title="Parthi AI Explainer :: Simple LLM - Input / Output ", page_icon="🤖", layout="wide")

st.markdown(
    """
<style>
.stApp { background: linear-gradient(180deg,#f7f9fc 0%, #ffffff 100%); }
.response-box { background: #fff; border-radius:10px; padding:14px; box-shadow: 0 6px 18px rgba(20,30,60,0.06); }
.header { display:flex; gap:12px; align-items:center; }
.big-title { font-size:1.5rem; font-weight:700; }
.small { color:#555; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------
# Sidebar - API & Settings
# ------------------------------
st.sidebar.header("🔐 API & Settings")

# default prompt stored in session if not present
DEFAULT_PROMPT = "Explain what is AI Agentics in simple terms. Make it fun and easy to understand. Use examples."

if "prompt_input" not in st.session_state:
    st.session_state["prompt_input"] = DEFAULT_PROMPT

if "history" not in st.session_state:
    st.session_state["history"] = []

api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Your OpenAI API key (sk-...)")
model_choice = st.sidebar.selectbox(
    "Model",
    options=["gpt-4o-mini", "gpt-5-mini", "gpt-4o"],
    index=0,
    help="Choose model (depends on availability in your account).",
)
temp = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, step=0.05, value=0.5)
st.sidebar.markdown("---")
st.sidebar.write("Tip: lower temperature → factual. higher → creative.")

# ------------------------------
# Layout: quick examples first (so they can update session state), then form
# ------------------------------
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div class="header"><div class="big-title">🤖 Parthi AI Explainer</div></div>', unsafe_allow_html=True)
    st.markdown("Make explanations simple, fun, and easy to understand.")

    # Quick example buttons placed BEFORE creating the form widget
    st.write("**Quick Examples**")
    ex1, ex2, ex3 = st.columns(3)
    if ex1.button("AI Agentics"):
        st.session_state["prompt_input"] = "Explain what is AI Agentics in simple terms. Make it fun and easy to understand. Use examples."
        safe_rerun()
    if ex2.button("What is RAG?"):
        st.session_state["prompt_input"] = "Explain Retrieval-Augmented Generation (RAG) in simple terms and give a use-case example."
        safe_rerun()
    if ex3.button("Invoke vs Stream"):
        st.session_state["prompt_input"] = "What's the difference between invoke() and stream() in LangChain? Explain in 3 bullets."
        safe_rerun()

    # Now create the form (text_area uses value= from session_state)
    with st.form("prompt_form"):
        prompt_value = st.text_area(
            "📝 Enter your question or topic",
            value=st.session_state.get("prompt_input", DEFAULT_PROMPT),
            height=160,
            placeholder="Type your prompt here...",
        )
        submit = st.form_submit_button("Generate Answer")

    # placeholders for response & tools
    response_container = st.empty()
    tools_container = st.empty()

with col2:
    st.markdown("### 🕘 History")
    # show last 6 history items
    for item in reversed(st.session_state["history"][-6:]):
        time_label = item.get("time").strftime("%H:%M:%S") if item.get("time") else ""
        st.markdown(f"**{time_label}** — {item.get('prompt')[:60]}...")
        st.caption(item.get("answer")[:140] + ("..." if len(item.get("answer")) > 140 else ""))
    if st.button("Clear History"):
        st.session_state["history"] = []
        safe_rerun()

# ------------------------------
# Async model call
# ------------------------------
async def get_answer_async(prompt_text: str, model_name: str, temperature: float):
    chat = OpenAI(model=model_name, temperature=temperature)
    response = await chat.ainvoke(prompt_text)
    return response

# ------------------------------
# Handle form submit
# ------------------------------
if submit:
    # Persist latest prompt into session_state so history and examples persist
    st.session_state["prompt_input"] = prompt_value

    if not api_key:
        st.sidebar.error("Please enter your OpenAI API Key in the sidebar.")
        st.stop()

    # set API key for the process
    os.environ["OPENAI_API_KEY"] = api_key

    with st.spinner("Generating answer..."):
        try:
            # call the LLM asynchronously
            response = asyncio.run(get_answer_async(st.session_state["prompt_input"], model_choice, temp))
            text = extract_text(response)

            # save to history
            st.session_state["history"].append({"time": datetime.datetime.now(), "prompt": st.session_state["prompt_input"], "answer": text})

            # show result
            with response_container.container():
                st.markdown('<div class="response-box">', unsafe_allow_html=True)
                st.markdown("### ✔ Here's your answer")
                st.markdown(text)
                st.markdown("</div>", unsafe_allow_html=True)

            # tools: download + copy
            with tools_container.container():
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.download_button("📥 Download Answer", data=text, file_name="ai_answer.txt", mime="text/plain")
                with c2:
                    st.text_area("Copy answer", value=text, height=120)

        except Exception as e:
            st.error(f"🚨 Error while calling model: {e}")
            with st.expander("Error details (expand)"):
                st.write(e)
