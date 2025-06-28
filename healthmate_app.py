import streamlit as st
from langchain_vectorstore import get_vectorstore
from langchain_chain import get_chain

st.set_page_config(page_title="HealthMate", page_icon="⚕️", layout="wide")

st.markdown(
    """
    <style>
    .chat-user {
        background-color: #0084ff;
        color: white;
        padding: 12px 18px;
        border-radius: 15px;
        max-width: 70%;
        font-weight: 600;
        margin-left: auto;
        margin-bottom: 10px;
    }
    .chat-bot {
        background-color: #f0f0f0;
        color: #222;
        padding: 12px 18px;
        border-radius: 15px;
        max-width: 70%;
        margin-bottom: 15px;
        white-space: pre-wrap;
    }
    .source-doc {
        background-color: #e8f0fe;
        border-left: 4px solid #4285f4;
        padding: 10px 15px;
        margin: 5px 0 20px 0;
        font-size: 0.9em;
        color: #444;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("HealthMate 🩺 - Advanced Medical Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

def clear_history():
    st.session_state.history = []

st.sidebar.button("Clear Conversation", on_click=clear_history)

vectorstore = get_vectorstore()
chain = get_chain(vectorstore)

def generate_response(user_input):
    inputs = {
        "question": user_input,
        "chat_history": st.session_state.history,
    }
    try:
        result = chain.invoke(inputs)
        answer = result["answer"]
        source_docs = result.get("source_documents", [])
        st.session_state.history.append((user_input, answer))
        return answer, source_docs
    except Exception as e:
        return f"⚠️ Error generating response: {e}", []

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Enter your medical question or symptoms:", key="input_text")
    submit_button = st.form_submit_button(label="Send")

if submit_button and user_input.strip():
    with st.spinner("HealthMate is thinking..."):
        response, sources = generate_response(user_input)

    st.markdown(f'<div class="chat-user">{user_input}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-bot">{response}</div>', unsafe_allow_html=True)

    if sources:
        st.markdown("<b>Source Documents:</b>", unsafe_allow_html=True)
        for i, doc in enumerate(sources):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            meta = doc.metadata if hasattr(doc, 'metadata') else {}
            url = meta.get('source', '') or meta.get('url', '') or ''
            url_display = f'<a href="{url}" target="_blank">{url}</a>' if url else ''
            st.markdown(
                f'<div class="source-doc">Doc {i+1}: {content[:300]}...<br>{url_display}</div>', 
                unsafe_allow_html=True
            )

if st.session_state.history:
    st.markdown("---")
    st.markdown("### Conversation History")

    for i, (q, a) in enumerate(st.session_state.history):
        st.markdown(f'<div class="chat-user">Q{i+1}: {q}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bot">A{i+1}: {a}</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    """
    ⚠️ **Disclaimer:** HealthMate provides informational answers based on publicly available medical FAQs.
    It is NOT a substitute for professional medical advice, diagnosis, or treatment.
    Always consult a healthcare professional for your health concerns.
    """
)
