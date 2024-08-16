# # import streamlit as st
# # import os
# # from langchain_groq import ChatGroq
# # from langchain_openai import OpenAIEmbeddings
# # from langchain_community.embeddings import OllamaEmbeddings
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain.chains.combine_documents import create_stuff_documents_chain
# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain.chains import create_retrieval_chain
# # from langchain_community.vectorstores import FAISS
# # from langchain_community.document_loaders import PyPDFDirectoryLoader
# # import openai

# # from dotenv import load_dotenv
# # load_dotenv()
# # ## load the GROQ API Key
# # os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
# # os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

# # groq_api_key=os.getenv("GROQ_API_KEY")

# # llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

# # prompt=ChatPromptTemplate.from_template(
# #     """
# #     Answer the questions based on the provided context only.
# #     Please provide the most accurate respone based on the question
# #     <context>
# #     {context}
# #     <context>
# #     Question:{input}

# #     """

# # )

# # def create_vector_embedding():
# #     if "vectors" not in st.session_state:
# #         st.session_state.embeddings=OpenAIEmbeddings()
# #         st.session_state.loader=PyPDFDirectoryLoader("research_papers") ## Data Ingestion step
# #         st.session_state.docs=st.session_state.loader.load() ## Document Loading
# #         st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
# #         st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
# #         st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
# # st.title("RAG Document Q&A With Groq And Lama3")

# # user_prompt=st.text_input("Enter your query from the research paper")

# # if st.button("Document Embedding"):
# #     create_vector_embedding()
# #     st.write("Vector Database is ready")

# # import time

# # if user_prompt:
# #     document_chain=create_stuff_documents_chain(llm,prompt)
# #     retriever=st.session_state.vectors.as_retriever()
# #     retrieval_chain=create_retrieval_chain(retriever,document_chain)

# #     start=time.process_time()
# #     response=retrieval_chain.invoke({'input':user_prompt})
# #     print(f"Response time :{time.process_time()-start}")

# #     st.write(response['answer'])

# #     ## With a streamlit expander
# #     with st.expander("Document similarity Search"):
# #         for i,doc in enumerate(response['context']):
# #             st.write(doc.page_content)
# #             st.write('------------------------')







# import os
# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.prompts import PromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# import time

# load_dotenv()

# # Load the GROQ and Google API key from the .env file
# groq_api_key = os.getenv("GROQ_API_KEY")
# os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# # Initialize the language model (llm)
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")

# # Function to create vector embedding
# def vector_embedding():
#     if "vectors" not in st.session_state:
#         with st.spinner("Creating vector store..."):
#             try:
#                 st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#                 st.session_state.loader = PyPDFDirectoryLoader("./us_census")
#                 st.session_state.docs = st.session_state.loader.load()
#                 st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#                 st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

#                 # Ensure that documents are not empty and embeddings are correctly initialized
#                 if not st.session_state.final_documents:
#                     st.error("No documents were loaded or split correctly.")
#                     return

#                 if len(st.session_state.final_documents) == 0:
#                     st.error("No documents found in the provided directory.")
#                     return

#                 # Creating FAISS vector store
#                 st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
#                 st.sidebar.success("Vector Store is ready!")

#             except Exception as e:
#                 st.error(f"Error creating FAISS vector store: {e}")

# # Function to generate document insights (Placeholder for actual implementation)
# def generate_insights(doc_content):
#     # Implement or use an existing summarizer model
#     return "Summary of the document..."

# # Sidebar for settings and actions
# st.sidebar.title("Settings & Actions")
# uploaded_files = st.sidebar.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True, key="upload")

# if st.sidebar.button("Create Vector Store", use_container_width=True):
#     if uploaded_files:
#         vector_embedding()
#     else:
#         st.sidebar.error("Please upload PDF files before creating the vector store.")

# # Main app content
# st.title("📄 Gemma Model Document Q&A")

# # Create a standard PromptTemplate
# prompt_template = PromptTemplate(
#     input_variables=["context", "input"],
#     template="""
#     Answer the questions based on the provided context only.
#     Please provide the most accurate response based on the question.
#     <context>
#     {context}
#     </context>
#     Questions: {input}
#     """
# )

# # Custom CSS for improved styling
# st.markdown("""
# <style>
#     .stButton>button {
#         background-color: #007bff;
#         color: #fff;
#         border-radius: 5px;
#         padding: 10px 20px;
#         font-size: 16px;
#         margin-top: 10px;
#     }
#     .stButton>button:hover {
#         background-color: #0056b3;
#     }
#     .stTextInput>div>input {
#         border: 2px solid #007bff;
#         border-radius: 5px;
#         padding: 10px;
#     }
#     .stDownloadButton>button {
#         background-color: #28a745;
#         color: #fff;
#         border-radius: 5px;
#         padding: 10px 20px;
#         font-size: 16px;
#         margin-top: 10px;
#     }
#     .stDownloadButton>button:hover {
#         background-color: #218838;
#     }
#     .stSpinner>div {
#         font-size: 18px;
#         font-weight: bold;
#     }
#     .stMarkdown {
#         margin-top: 20px;
#     }
#     .stText {
#         margin-top: 10px;
#     }
#     .query-history {
#         margin-top: 30px;
#         padding: 10px;
#         background-color: #f8f9fa;
#         border-radius: 5px;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Section: User Input
# st.subheader("Ask a Question")
# prompt1 = st.text_input("What do you want to ask from the documents?", placeholder="Enter your question here...")

# # Section: Output
# if prompt1:
#     if "vectors" in st.session_state:
#         with st.spinner("Processing your request..."):
#             document_chain = create_stuff_documents_chain(llm, prompt_template)
#             retriever = st.session_state.vectors.as_retriever()
#             retrieval_chain = create_retrieval_chain(retriever, document_chain)

#             start = time.process_time()
#             response = retrieval_chain.invoke({'input': prompt1})
#             end = time.process_time()

#             st.success(f"Response generated in {end - start:.2f} seconds")

#             st.subheader("Answer:")
#             st.write(response.get('answer', "No answer found."))

#             if "context" in response:
#                 total_pages = len(response["context"]) // 5 + 1
#                 page_number = st.number_input("Page number", min_value=1, max_value=total_pages, value=1, step=1)
                
#                 def display_documents(docs, page=1, docs_per_page=5):
#                     start = (page - 1) * docs_per_page
#                     end = start + docs_per_page
#                     for i, doc in enumerate(docs[start:end], start=start + 1):
#                         st.markdown(f"**Document {i}:**")
#                         st.write(doc.page_content)
#                         st.write("**Summary:**", generate_insights(doc.page_content))
#                         st.write("---")
                
#                 display_documents(response["context"], page=page_number)

#                 st.download_button(
#                     label="Download Answer",
#                     data=response['answer'],
#                     file_name='answer.txt',
#                     mime='text/plain',
#                 )
#             else:
#                 st.error("No relevant documents found.")
#     else:
#         st.error("Vector store not initialized. Please create the vector store first using the sidebar.")
# else:
#     st.info("Please enter a question above to get started.")

# # Query History
# if "query_history" not in st.session_state:
#     st.session_state.query_history = []

# if prompt1:
#     st.session_state.query_history.append(prompt1)

# st.subheader("Query History")
# st.markdown('<div class="query-history">', unsafe_allow_html=True)
# for query in st.session_state.query_history[-5:]: # Display the last 5 queries
#     st.write(query)
# st.markdown('</div>', unsafe_allow_html=True)


import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()