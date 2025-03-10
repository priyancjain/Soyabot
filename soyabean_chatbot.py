from flask import Flask, render_template, request, jsonify
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

app = Flask(__name__)

DB_FAISS_PATH = r"/Users/priyanshijain/project/Soyabot/vectorstore/db_faiss"

def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm():
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        token=os.environ.get("HF_TOKEN"),
        temperature=0.5,
        task="text-generation",
        model_kwargs={"max_length": 256}
    )

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')

    if not user_input:
        return jsonify({'error': 'Invalid input'}), 400

    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            return jsonify({'error': "Failed to load the vector store"}), 500

        CUSTOM_PROMPT_TEMPLATE = """
        You are the Soyabean crop expert who can answer in Hindi and English. Answer queries in detail but avoid unnecessary information.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk, please.
        """

        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
            return_source_documents=False,
            chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
        )

        response = qa_chain.invoke({'query': user_input})
        result = response["result"]

        # 🔹 Format the response (Bold headings & bullet points)
        # Format the response (Bold headings & bullet points)
        # Format the response (Bold headings & bullet points)
        formatted_result = (
        ""
        + result.replace(". ", ".\n- ") + "\n\n"
        "---\n"
        "\n"
        )


        return jsonify({'response': formatted_result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)



# import os
# from flask import Flask, request, jsonify
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint

# app = Flask(__name__)

# DB_FAISS_PATH = r"/Users/priyanshijain/project/Soyabot/vectorstore/db_faiss"

# # Load vector store (FAISS)
# def get_vectorstore():
#     embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     return db

# # Custom Prompt Template
# def set_custom_prompt(custom_prompt_template):
#     return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# # Load LLM
# def load_llm(huggingface_repo_id, HF_TOKEN):
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         token=HF_TOKEN,
#         temperature=0.5,
#         task="text-generation",
#         model_kwargs={"max_length": 256}
#     )
#     return llm

# # Define the chatbot API endpoint
# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.json  # Get JSON input
#     prompt = data.get("query", "").strip()

#     if not prompt:
#         return jsonify({"error": "Query is required"}), 400

#     CUSTOM_PROMPT_TEMPLATE = """
#         You are a Soybean crop expert who can answer in Hindi and English. 
#         Answer queries in detail and avoid unnecessary information.

#         Context: {context}
#         Question: {question}

#         Start the answer directly. No small talk please.
#     """

#     HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
#     HF_TOKEN = os.environ.get("HF_TOKEN")

#     try:
#         vectorstore = get_vectorstore()
#         if vectorstore is None:
#             return jsonify({"error": "Failed to load the vector store"}), 500

#         qa_chain = RetrievalQA.from_chain_type(
#             llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
#             chain_type="stuff",
#             retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
#             return_source_documents=False,
#             chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#         )

#         response = qa_chain.invoke({'query': prompt})
#         result = response["result"]

#         return jsonify({"answer": result})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)


# import os
# import streamlit as st

# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA

# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint

# ## Uncomment the following files if you're not using pipenv as your virtual environment manager
# #from dotenv import load_dotenv, find_dotenv
# #load_dotenv(find_dotenv())


# DB_FAISS_PATH=r"/Users/priyanshijain/project/Soyabot/vectorstore/db_faiss"
# @st.cache_resource
# def get_vectorstore():
#     embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     return db


# def set_custom_prompt(custom_prompt_template):
#     prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt


# def load_llm(huggingface_repo_id, HF_TOKEN):
#     llm=HuggingFaceEndpoint(
#          repo_id=huggingface_repo_id,
#         token=HF_TOKEN,  # API Token
#         temperature=0.5,  # Set temperature directly
#           # Move outside model_kwargs
#         task="text-generation",
#         model_kwargs={ 
#             "max_length": 256  # Keep only max_length here
#         }
#     )
#     return llm


# def main():
#     st.title("🌱 Soyabean Chatbot")

#     if 'messages' not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         st.chat_message(message['role']).markdown(message['content'])

#     prompt = st.chat_input(" Ask me anything about Soybean farming...")

#     if prompt:
#         st.chat_message('user').markdown(prompt)
#         st.session_state.messages.append({'role': 'user', 'content': prompt})

       

#         CUSTOM_PROMPT_TEMPLATE = """
#              you are the Soyabean crop expert who can answer in hindi and english language , so answer the queries of the user in detail ,
#         and make sure you do not give unnecessay information to the question.

#         Context: {context}
#         Question: {question}

#         Start the answer directly. No small talk please.  
#             """

#         HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
#         HF_TOKEN = os.environ.get("HF_TOKEN")

#         try:
#             vectorstore = get_vectorstore()
#             if vectorstore is None:
#                 st.error("Failed to load the vector store")
#                 return

#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
#                 chain_type="stuff",
#                 retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
#                 return_source_documents=False,
#                 chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#             )

#             response = qa_chain.invoke({'query': prompt})

#             # Extract only the result (answer)
#             result = response["result"]

#             formatted_result = f"""
#             ### **📝 Answer:**  
#             {result}  

#             ---
#              **Need more details? Ask a follow-up question!**
#             """

#             # Display only the result (text answer) without source documents
#             st.chat_message('assistant').markdown(result)
#             st.session_state.messages.append({'role': 'assistant', 'content': result})

#         except Exception as e:
#             st.error(f"Error: {str(e)}")


# if __name__ == "__main__":
#     main()