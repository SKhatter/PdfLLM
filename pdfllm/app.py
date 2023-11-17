import streamlit as st 
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from PyPDF2 import PdfFileReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback 
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter



# Sidebar contents
with st.sidebar:
    st.title("LLM Chat App")
    st.markdown('''
        ## About 
                This app is LLM-powered 
                chatbot, created by,
                _skhatter. 
                
                **Tech Stack**
                OpenAI models
                Langchain
                StreamLit 
        ''')
    add_vertical_space(15)
    
    st.write("Made with LOVE")



def main():
    st.write("Chat with the PDF!")
    load_dotenv()

    #upload your pdf
    pdf = st.file_uploader("Upload Your pdf", type ='pdf')
    

    if pdf:
        st.write(pdf.name)
        pdf_reader = PdfReader(pdf)
        #pdf_reader = PdfFileReader(pdf)
        #st.write(pdf_reader)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size= 200,
            chunk_overlap=30, #chunk overlap is important
            length_function=len
        )
        #loader = TextLoader(text)
        #documents = loader.load()

        #char_text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        #docs = char_text_splitter.split_documents(documents)

        chunks = text_splitter.split_text(text=text)
        #chunks = text_splitter.split_documents(docs)

        #st.write(chunks)

        #embeddings
       
        store_name = pdf.name[:-4]
        ###
        ###if os.path.exists(f"{store_name}.pk1"):
           ###         with open(f"{store_name}.pk1", "rb") as f:
        ###                vectorstore = pickle.load(f)
        ###            st.write('Embeddings Loaded from the disk')
        ###        else:
        
         
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks,embeddings)
        #print("hiiiii", vectorstore.docstore._dict)
        
        #with open(f"{store_name}.pk1", "wb") as f:
            #pickle.dump(vectorstore.docstore._dict, f)

        st.write('Embeddings Computation Completed.')

        # Accept User Questions 
        query = st.text_input("Ask Questions about your pdf file?")
        st.write(query)
        if query:
            #print('Hey you', type(vectorstore), list(vectorstore.keys()), list(vectorstore.values()))
            docs = vectorstore.similarity_search(query=query, k=3) #context window, the number of relevant k
            llm = OpenAI(temperature=0)
            #llm = OpenAI(model_name = 'gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:

                response = chain.run(input_documents=docs, question=query)
                print(cb)
                st.write(response)

            #st.write(docs)

        


if __name__ == '__main__':
    main()