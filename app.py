import streamlit as st
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import time
import os

from lib.multi_modal_rag import (
    extract_pdf_elements,
    categorize_elements,
)

import dotenv
dotenv.load_dotenv()

SAVE_DIR = "uploaded_files/"


st.title("DexDLab GPT")

with st.sidebar:
    uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
    process = st.button("Process")
if process:
    st.title("haha fun")

    for uploaded_file in uploaded_files:

        st.info(uploaded_file)

        file_path = os.path.join(SAVE_DIR, uploaded_file.name)
    
        # 파일 저장
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 요소 추출
        raw_pdf_elements = extract_pdf_elements(SAVE_DIR, uploaded_file.name)

        # 텍스트, 테이블 추출
        texts, tables = categorize_elements(raw_pdf_elements)

        st.info(texts)
        st.info(tables)


# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # LLM
        llm = ChatOpenAI()

        # Prompt
        llm_prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "You are a nice chatbot having a conversation with a human."
                ),
                # The `variable_name` here is what must align with memory
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )

        # Notice that we `return_messages=True` to fit into the MessagesPlaceholder
        # Notice that `"chat_history"` aligns with the MessagesPlaceholder name
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation = LLMChain(llm=llm, prompt=llm_prompt, verbose=True, memory=memory)

        result = conversation({"question": prompt})
        for chunk in result['text'].split():
            full_response += chunk + " "
            time.sleep(0.05)

            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})