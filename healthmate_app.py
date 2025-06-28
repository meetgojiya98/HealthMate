import streamlit as st
from langchain_vectorstore import get_vectorstore
from langchain_chain import get_chain

st.set_page_config(page_title="HealthMate", page_icon="⚕️", layout="centered")

# Inject custom CSS for chat bubbles & avatars
st.markdown(
    """
    <style>
    /* Container */
    .chat-container {
        max-width: 700px;
        margin: 20px auto;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* User bubble */
    .user-msg {
        background-color: #0078d4;
        color: white;
        padding: 12px 16px;
        border-radius: 20px 20px 0 20px;
        max-width: 75%;
        margin-left: auto;
        margin-bottom: 10px;
        position: relative;
        clear: both;
    }
    .user-avatar {
        float: right;
        margin-left: 8px;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: #0078d4 url('https://cdn-icons-png.flaticon.com/512/147/147144.png') no-repeat center;
        background-size: cover;
    }

    /* HealthMate bubble */
    .bot-msg {
        background-color: #e1f0ff;
        color: #222;
        padding: 12px 16px;
        border-radius: 20px 20px 20px 0;
        max-width: 75%;
        margin-right: auto;
        margin-bottom: 10px;
        position: relative;
        clear: both;
    }
    .bot-avatar {
        float: left;
        margin-right: 8px;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: #005a9e url('https://cdn-icons-png.flaticon.com/512/1998/1998611.png') no-repeat center;
        background-size: cover;
    }

    /* Sources */
    .sources {
        font-size: 14px;
        margin-top: -8px;
        margin-bottom: 12px;
        color: #555;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="chat-container">
    <h1 style="text-align:center; color:#004a99;">HealthMate 🩺</h1>
    <p style="text-align:center; color:#666; font-size:18px;">Your AI-powered medical assistant</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []

def clear_history():
    st.session_state.history = []

st.sidebar.markdown("## Controls")
st.sidebar.button("Clear Conversation", on_click=clear_history)

vectorstore = get_vectorstore()
chain = get_chain(vectorstore)

def generate_response(user_input):
    inputs = {"question": user_input}
    try:
        result = chain.invoke(inputs)
        answer = result.get("answer", "Sorry, no answer found.")
        sources = result.get("source_documents", [])
        st.session_state.history.append((user_input, answer, sources))
        return answer, sources
    except Exception as e:
        return f"⚠️ Error generating response: {e}", []

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask your medical question or describe symptoms:")
    submit_button = st.form_submit_button("Send")

if submit_button and user_input.strip():
    with st.spinner("HealthMate is processing..."):
        response, source_docs = generate_response(user_input)

    # Display User message bubble
    st.markdown(
        f"""
        <div class="chat-container">
            <div class="user-msg">{user_input}</div>
            <div class="user-avatar"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Display HealthMate message bubble
    st.markdown(
        f"""
        <div class="chat-container">
            <div class="bot-avatar"></div>
            <div class="bot-msg">{response}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if source_docs:
        st.markdown("<div class='sources'><strong>Sources:</strong></div>", unsafe_allow_html=True)
        for i, doc in enumerate(source_docs):
            source = getattr(doc.metadata, "source", None) if hasattr(doc, "metadata") else None
            snippet = doc.page_content[:300] + "..." if hasattr(doc, "page_content") else ""
            label = source if source else f"Source snippet {i+1}"
            with st.expander(label):
                if source:
                    st.markdown(f"[Link]({source})")
                st.markdown(snippet)

if st.session_state.history:
    st.markdown("---")
    st.markdown("## Conversation History")
    for i, (q, a, sources) in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")
        if sources:
            with st.expander(f"Sources for A{i}"):
                for j, doc in enumerate(sources):
                    source = getattr(doc.metadata, "source", None) if hasattr(doc, "metadata") else None
                    snippet = doc.page_content[:300] + "..." if hasattr(doc, "page_content") else ""
                    label = source if source else f"Source snippet {j+1}"
                    with st.expander(label):
                        if source:
                            st.markdown(f"[Link]({source})")
                        st.markdown(snippet)

st.markdown(
    """
    <hr>
    <p style="font-size:12px; color:#999; text-align:center;">
    ⚠️ <strong>Disclaimer:</strong> HealthMate provides informational answers based on publicly available medical FAQs.<br>
    It is NOT a substitute for professional medical advice, diagnosis, or treatment.<br>
    Always consult a healthcare professional for your health concerns.
    </p>
    """,
    unsafe_allow_html=True,
)
