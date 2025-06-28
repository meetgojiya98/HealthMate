from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

def get_chain(vectorstore):
    # Setup HF GPT2 pipeline LLM (or replace with preferred model)
    pipe = pipeline(
        "text-generation",
        model="gpt2",
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # ConversationalRetrievalChain uses default keys: question, chat_history
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False,  # Set True if you want to display source docs
    )

    return chain
