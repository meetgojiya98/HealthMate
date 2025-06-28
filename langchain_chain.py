from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

INDEX_DIR = "embeddings/faiss_index"

def get_chain(vectorstore):
    # Setup HuggingFace text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model="gpt2",
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # Conversation memory for multi-turn
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Prompt template uses 'question' instead of 'query'
    prompt_template = PromptTemplate(
        input_variables=["question", "context", "chat_history"],
        template=(
            "You are a helpful medical assistant.\n"
            "Chat history:\n{chat_history}\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        output_key="answer",
    )
    return chain
