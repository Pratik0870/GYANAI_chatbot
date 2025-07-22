# 🤖 GyaanAI – Gemini-Style PDF Chatbot

GyaanAI is a smart chatbot that can read PDF files, answer your questions, and summarize the content using **LangChain** and **Groq's LLaMA3 models**.

## 📦 Features
- Upload any PDF and ask questions from it
- Summarize PDF content with AI
- Powered by `llama3-70b-8192` via [Groq Cloud](https://console.groq.com)

## 🚀 How to Run

```bash
pip install -r requirements.txt
cp .env.example .env   # Then add your Groq API key inside .env
streamlit run app.py
