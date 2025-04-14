import pandas as pd
from langchain_openai import ChatOpenAI
import openai
from langchain_core.prompts import ChatPromptTemplate
from retrieval import create_vector_store, create_retriever
from embedding import JAIEmbeddings, initialize_jai_client
from data_processing import preprocess_dataframe

from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableMap
from langchain_voyageai import VoyageAIEmbeddings

import streamlit as st
from langfuse.callback import CallbackHandler

api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API Key to continue.")
    st.stop()

print(api_key)

def initialize_llm():
    llm_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=500, timeout=10, max_retries=2, streaming=False, api_key=api_key)
    return llm_model

def create_retrieval_chain(retriever, llm):
    """Create a retrieval-augmented generation chain using LCEL."""
    
    # Prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
            You are a friendly female assistant who answers questions about series.
            Respond to user questions using only the information provided in the Context, ensuring all answers are accurate and based on that data only.
            Do not guess or add any information that is not in the Context.
            Include a link to the series in your response.
            Respond in Thai only.
        """),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    # LCEL retrieval + prompt + LLM + output parser
    chain = (
        RunnableMap({
            "context": lambda x: "\n\n".join([doc.page_content for doc in retriever.invoke(x["question"])]),
            "question": lambda x: x["question"],
        })
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return chain

def get_chat_history_text(chat_history, limit=2):
    return "\n".join(
        [f"ผู้ใช้: {msg['content']}" if msg["role"] == "user" else f"ผู้ช่วย: {msg['content']}" 
         for msg in chat_history[-limit*2:]]
    )

def chatbot_response(chain, user_input, llm, langfuse_handler):
    """ answer user's query and provide follow-up question if needed """
     # retireve chat history
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_history = st.session_state.messages

    # retrieve previous queries
    previous_queries = [msg["content"] for msg in chat_history if msg["role"] == "user"]
    # Combine with latest input
    query = " ".join(previous_queries[-2:] + [user_input])

    print(f"[DEBUG] Final combined query: {query}")

    response = chain.invoke({"question": query}, config={"callbacks": [langfuse_handler]})

    # save query and answer to the memory
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})

    # check if the query is too broad
    is_broad, followup_question = check_and_generate_followup(llm, chat_history, user_input)
    if is_broad and followup_question:
        return f"{response}\n\n{followup_question}"

    return response

def check_and_generate_followup(llm, chat_history, user_input):
    """
    Checks if the user's question is too broad.
    If so, generates a follow-up question to clarify.
    """
    previous_queries = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
    print(f"[DEBUG] Final previous_queries: {previous_queries}")

    followup_prompt = f"""
    You are an AI assistant that checks if the user's question is specific enough. 
    If not, generate a follow-up question to help narrow down their intent.

    Chat History:
    {get_chat_history_text(chat_history)}

    Latest user question: "{user_input}"

    Guidelines:
    - A question is too broad if it lacks key information such as a series title, genre, director, cast, or other specific attributes.
    - If the question is too vague, suggest a follow-up question that asks for more detail (e.g., title, genre, cast, synopsis, etc.).
    - Be natural and concise. Don't repeat what the user already asked.
    - Do **not** ask about platforms or where to watch.
    - Respond in **Thai only**.
    - If the original question is already specific enough, return **nothing**.

    Follow-up Question (if needed):
    """

    followup_question = llm.invoke(followup_prompt).content.strip()

    if followup_question:
        return True, followup_question
    return False, None

def chat_loop(llm_for_chat, qa_chain, langfuse_handler):
    """Start chatting with user using memory and context."""
    st.title("Series Search Assistant")
    st.write("สวัสดีค่ะ! ฉันสามารถช่วยแนะนำซีรีส์หรือให้ข้อมูลเกี่ยวกับนักแสดง ผู้กำกับ และเรื่องย่อของซีรีส์ต่าง ๆ ได้ คุณกำลังมองหาซีรีส์แบบไหนอยู่คะ?")    

    # Initialize display_messages if it doesn't exist
    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []
    
    # Display all messages in the chat history
    for message in st.session_state.display_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    prompt = st.chat_input("คุณ:", key="chat_input")

    if prompt:
        # User exits the chat
        if prompt.lower() == "exit":
            st.session_state.messages.append({"role": "assistant", "content": "ขอบคุณค่ะ หวังว่าข้อมูลจะเป็นประโยชน์นะคะ!"})
            st.stop()  # Restart the app so the user sees the exit message

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        response = chatbot_response(qa_chain, prompt, llm_for_chat, langfuse_handler)
        
        # Display bot response
        with st.chat_message("assistant"):
            st.markdown(response)
            
        # Append to display messages instead of replacing them
        st.session_state.display_messages.append({"role": "user", "content": prompt})
        st.session_state.display_messages.append({"role": "assistant", "content": response})

        st.rerun()  # Refresh chat UI to maintain conversation history


if __name__ == "__main__":
    langfuse_handler = CallbackHandler()
    
    # Load data
    df = pd.read_csv("test_data.csv")
    df = preprocess_dataframe(df)
    texts = df["data"].tolist()

    # Initialize LLM, client, embeddings
    llm_for_chat = initialize_llm()
    client = initialize_jai_client()
    jai_embedding = JAIEmbeddings(client, model_name="jai-emb-passage")
    voyage = VoyageAIEmbeddings(voyage_api_key="pa-lTzTRUJQkxetNEBkwD8oq59W0vlcFePpZ3kPAv-ZP6p", model="voyage-3")
    # Create vector store and retriever
    vector_store = create_vector_store(voyage, texts)
    
    ensemble_retriever = create_retriever(texts, vector_store)

    qa_chain = create_retrieval_chain(ensemble_retriever, llm_for_chat)

    # Start app
    chat_loop(llm_for_chat, qa_chain, langfuse_handler)
