import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma  
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
#HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")


# Get your Hugging Face token from environment variable
# Add this to your .env file
# we dont need to use Hugging Face token for free models, beacuse we are creating embedding locally using sentance transformer 


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    # Using free Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # bge model can be used as well
        # model_name="BAAI/bge-small-en-v1.5", but this requires hugging face token  # Uncomment this line to use BGE embeddings
        # model_name="BAAI/bge-base-en-v1.5",  # Uncomment this line to use BGE embeddings
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
        # Ensure embeddings are normalized
    )
    
    vector_store = Chroma.from_texts(
        text_chunks,
        embedding=embeddings,
        persist_directory="chromadb_index"
    )
    vector_store.persist()


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGroq(
    api_key=groq_api_key,
    model_name="Gemma2-9b-It",  # You can change this to llama3 or gemma
    temperature=0.2
    )

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    # Using free Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # bge model can be used as well
        # model_name="BAAI/bge-small-en-v1.5", but this requires hugging face token  # Uncomment this line to use BGE embeddings
        # model_name="BAAI/bge-base-en-v1.5",  # Uncomment this line to use BGE embeddings
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
       
         # Ensure embeddings are normalized
    )
    
    new_db = Chroma(persist_directory="chromadb_index", embedding_function=embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config(page_title="Chat PDF")

    st.header("Chat with PDF using Groq💁")

   

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()