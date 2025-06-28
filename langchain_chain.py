from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

INDEX_DIR = "embeddings/faiss_index"

def get_chain(vectorstore):
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=512,
        temperature=0,
        do_sample=False,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    prompt_template = PromptTemplate(
        input_variables=["question", "context", "chat_history"],
        template=(
            "You are a helpful medical assistant.\n"
            "Use the context below to answer the question clearly and in a detailed manner.\n"
            "If the answer is not in the context, say you don't know.\n\n"
            "Context:\n{context}\n\n"
            "Chat history:\n{chat_history}\n\n"
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
