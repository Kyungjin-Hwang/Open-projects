import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

api_key = st.secrets["OPENAI_API_KEY"]

def process_text(text):
    #CharacterTextSplitter을 사용하여 텍스트를 청크로 분할
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap= 200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    #임베딩 처리 (벡터 변환)
    embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002", api_key = api_key)
    documents = FAISS.from_texts(chunks, embeddings)
    return documents

def main():
    st.title("Resumen de PDF")
    st.divider()

    pdf = st.file_uploader("Sube un archivo de PDF", type = 'pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = "" #텍스트 변수에 pdf 저장
        for page in pdf_reader.pages:
            text += page.extract_text()

        documents = process_text(text)
        query = "Resuma el contenido del archivo PDF cargado en unas 3 a 5 oraciones, por favor."

        if query:
            docs = documents.similarity_search(query)
            model_name = "gpt-4-turbo"
            llm = ChatOpenAI(model_name=model_name, api_key=api_key, temperature = 0.1)
            chain = load_qa_chain(llm, chain_type='stuff')
            with get_openai_callback() as cost:
                response = chain.run(input_documents = docs, question = query)
                print(cost)
            st.subheader('Resumen del archivo')
            st.write(response)

if __name__ == '__main__':
    main()