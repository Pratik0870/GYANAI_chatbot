import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    temperature=0.3,
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192"  
)
parser = StrOutputParser()

qa_prompt = ChatPromptTemplate.from_template("""
You are GyaanAI – an intelligent assistant trained on PDFs. Read the context and answer the question clearly.

Context:
{context}

Question:
{question}

Answer:""")

summary_prompt = ChatPromptTemplate.from_template("""
Summarize the following educational content clearly in 3-4 lines:

{text}

Summary:
""")


qa_chain = qa_prompt | llm | parser
summary_chain = summary_prompt | llm | parser


def read_pdf(file):
    reader = PdfReader(file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])


def split_text(text, chunk_size=1600, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


st.set_page_config(page_title="📘 GyaanAI  Chatbot", layout="wide")
st.title("📘 GyaanAI – Smart PDF Chatbot (Groq + LangChain)")
st.markdown("Upload a PDF, ask questions from it, and generate summaries using `gemma-7b-it` by Groq.")


uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"], key="pdf_upload")

if uploaded_file:
    text = read_pdf(uploaded_file)
    chunks = split_text(text)

    st.subheader("💬 Ask a Question")
    question = st.text_input("What do you want to ask about this document?")

    if question:
        results = []
        with st.spinner("Thinking..."):
            for chunk in chunks[:3]:
                try:
                    answer = qa_chain.invoke({"context": chunk, "question": question})
                    results.append(answer)
                except Exception as e:
                    st.warning(f"Skipped a chunk due to error: {e}")
        if results:
            st.markdown("###  GyaanAI Answer")
            st.write("\n\n".join(results))
        else:
            st.error(" Could not generate an answer. Try rephrasing your question.")

    st.subheader("📌 Generate Summary")
    if st.button("Summarize First Few Pages"):
        with st.spinner("Summarizing..."):
            try:
                summary = summary_chain.invoke({"text": " ".join(chunks[:2])})
                st.markdown("### 📝 Summary")
                st.write(summary)
            except Exception as e:
                st.error(f"Failed to summarize: {e}")
