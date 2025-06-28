import streamlit as st
from langchain_vectorstore import get_vectorstore
from langchain_chain import get_chain

st.set_page_config(page_title="HealthMate", page_icon="⚕️")

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
        "question": user_input  # <-- key must match prompt_template
    }
    try:
        result = chain.invoke(inputs)
        answer = result.get("answer", "Sorry, no answer found.")
        sources = result.get("source_documents", [])
        st.session_state.history.append((user_input, answer, sources))
        return answer, sources
    except Exception as e:
        return f"⚠️ Error generating response: {e}", []

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Enter your medical question or symptoms:", key="input_text")
    submit_button = st.form_submit_button(label="Send")

if submit_button and user_input.strip():
    with st.spinner("HealthMate is thinking..."):
        response, source_docs = generate_response(user_input)
    st.markdown(f"**You:** {user_input}")
    st.markdown(f"**HealthMate:** {response}")

    if source_docs:
        st.markdown("**Sources:**")
        for doc in source_docs:
            source = getattr(doc.metadata, "source", None) if hasattr(doc, "metadata") else None
            snippet = doc.page_content[:200] + "..." if hasattr(doc, "page_content") else ""
            if source:
                st.markdown(f"- [{source}]({source})")
            else:
                st.markdown(f"- {snippet}")

if st.session_state.history:
    st.markdown("---")
    st.markdown("### Conversation History")
    for i, (q, a, sources) in enumerate(st.session_state.history):
        st.markdown(f"**Q{i+1}:** {q}")
        st.markdown(f"**A{i+1}:** {a}")
        if sources:
            st.markdown("**Sources:**")
            for doc in sources:
                source = getattr(doc.metadata, "source", None) if hasattr(doc, "metadata") else None
                snippet = doc.page_content[:200] + "..." if hasattr(doc, "page_content") else ""
                if source:
                    st.markdown(f"- [{source}]({source})")
                else:
                    st.markdown(f"- {snippet}")
        st.markdown("---")

st.markdown(
    """
    ⚠️ **Disclaimer:** HealthMate provides informational answers based on publicly available medical FAQs.
    It is NOT a substitute for professional medical advice, diagnosis, or treatment.
    Always consult a healthcare professional for your health concerns.
    """
)
