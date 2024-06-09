if ('RUN_MODE' in globals() or 'RUN_MODE' in locals()) and RUN_MODE == "CLOUD":
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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
    generate_text_summaries,
    generate_img_summaries,
    create_multi_vector_retriever,
    multi_modal_rag_chain,
)
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

import dotenv
dotenv.load_dotenv()

SAVE_DIR = "uploaded_files/"




st.title("DexDLab GPT")

rag_on = st.toggle("멀티모달 RAG 사용하기")


if rag_on:
    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf'],accept_multiple_files=True)
        if len(uploaded_files) > 0:
            process = st.button("업로드한 데이터 등록하기")
        else:
            process = None

        if process:
        # st.title("haha fun")


            for uploaded_file in uploaded_files:

                # st.info(uploaded_file)
                # st.info(uploaded_file)

                file_path = os.path.join(SAVE_DIR, uploaded_file.name)
            
                # 파일 저장
                

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 요소 추출
                st.info("[1/5] PDF에서 텍스트와 테이블을 추출중..")
                raw_pdf_elements = extract_pdf_elements(SAVE_DIR, uploaded_file.name)

                # 텍스트, 테이블 추출
                texts, tables = categorize_elements(raw_pdf_elements)

                # st.info(texts)
                # st.info(tables)

                # 선택사항: 텍스트에 대해 특정 토큰 크기 적용
                text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=4000, chunk_overlap=0  # 텍스트를 4000 토큰 크기로 분할, 중복 없음
                )
                joined_texts = " ".join(texts)  # 텍스트 결합
                texts_4k_token = text_splitter.split_text(joined_texts)
                # st.info(texts_4k_token)

                # 텍스트, 테이블 요약 가져오기
                st.info("[2/5] 텍스트, 테이블 요약을 생성중..")
                text_summaries, table_summaries = generate_text_summaries(
                    texts_4k_token, tables, summarize_texts=True
                )

                # 이미지 요약 실행
                st.info("[3/5] 이미지 요약 생성중")
                fg_path = "figures/"
                img_base64_list, image_summaries = generate_img_summaries(fg_path)

                # 요약을 색인화하기 위해 사용할 벡터 저장소
                vectorstore = Chroma(
                    collection_name="sample-rag-multi-modal", embedding_function=OpenAIEmbeddings()
                )

                # 검색기 생성
                st.info("[4/5] 검색기 생성")
                retriever_multi_vector_img = create_multi_vector_retriever(
                    vectorstore,
                    text_summaries,
                    texts,
                    table_summaries,
                    tables,
                    image_summaries,
                    img_base64_list,
                )

                # RAG 체인 생성
                st.info("[5/5] RAG 체인 생성")
                st.session_state.chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)

                st.info("등록한 문서에대한 RAG 준비가 완료되었습니다. 질문을 입력하세요.")

                # query = "몇건의 거래가 있었는지 알려줘"
                # st.info("질문중")
                # print(chain_multimodal_rag.invoke(query))
                # st.info(chain_multimodal_rag.invoke(query))





# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain_multimodal_rag" not in st.session_state:
    st.session_state.chain_multimodal_rag = None
    if rag_on: st.info("테스트할 PDF 파일을 업로드 하세요")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        
if prompt := st.chat_input("질문을 입력하세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    
    with st.chat_message("assistant"):
        
        if rag_on:
            message_placeholder = st.empty()
            if st.session_state.chain_multimodal_rag:
                full_response = st.session_state.chain_multimodal_rag.invoke(prompt)
            else:
                full_response = "멀티모드 RAG를 테스트할 파일을 먼저 등록하세요"
            message_placeholder.markdown(full_response)

            
        else:
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