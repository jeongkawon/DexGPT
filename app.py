import streamlit as st

print("hello")

st.title("Hello World")

with st.sidebar:
    uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
    process = st.button("Process")
if process:
    st.title("haha fun")