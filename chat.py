import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from retrieval import create_vector_store, create_retriever
from embedding import JAIEmbeddings, initialize_jai_client

import streamlit as st

from langfuse.callback import CallbackHandler

def initialize_llm():
    llm_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=500, timeout=None, max_retries=2, streaming=True)
    return llm_model

def create_qa_chain(ensemble_retriever, llm_for_chat):
    """ combine llm chain with ensemble triever"""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    custom_prompt = PromptTemplate.from_template("""
    You are a friendly female assistant who answers questions about series.
    Respond to user questions using only the information provided in the Context, ensuring all answers are accurate and based on that data only.
    Do not guess or add any information that is not in the Context.
    Include a link to the series in your response.
    Response naturally in Thai only

    Context:
    {context}

    Question: {question}

    Answer:
    """)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm_for_chat,
        retriever=ensemble_retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt} 
    )
    return qa_chain, memory

def check_and_generate_followup(llm, chat_history, user_input):
    """
    Checks if the user's question is too broad.
    If so, generates a follow-up question to clarify.
    Returns:
        (is_broad: bool, followup_question: str or None)
    """
    previous_queries = [msg.content for msg in chat_history]

    broad_question_prompt = f"""
    You are an AI that helps determine whether a user's question is specific enough.
    The database includes the following fields:

    Title
    Director
    Cast
    Synopsis
    Genre
    Duration
    Number of episodes
    Related links

    Here are the last few relevant user questions:
    {previous_queries[-3:]}

    Latest user question: "{user_input}"

    Evaluation Criteria:
    The question is too broad (respond YES) if it includes fewer than 2 fields and no title, or if it is too vague.
    The question is specific enough (respond NO) if it includes at least 2 relevant fields or already has a title.

    Examples:
    "Recommend a series" → YES
    "Are there any detective series directed by Christopher Nolan?" → NO
    "Who stars in Reply 1988?" → NO

    Respond using only "YES" or "NO"
    """

    is_broad = llm.invoke(broad_question_prompt).content.strip().upper() == "YES"

    if is_broad:
        followup_prompt = f"""
        Here is the chat history:
        {chat_history}

        The user asked: "{user_input}"

        This question is not specific enough.
        Do **not** ask about platforms.

        Please generate a follow-up question that helps retrieve more relevant information from the database.

        The follow-up question should guide the user to specify one or more of the following:
        - Title
        - Director
        - Cast
        - Synopsis
        - Genre
        - Duration
        - Number of Episodes
        - Related Links

        Be natural, concise, and do not repeat what the user already provided.

        Response in Thai only.

        Follow-up Question:
        """

        followup_question = llm.invoke(followup_prompt).content.strip()
        return True, followup_question

    return False, None


def chatbot_response(llm_for_chat, qa_chain, memory, user_input, langfuse_handler):
    """ answer user's query and provide follow-up question if needed """
     # retireve chat history

    chat_history = memory.load_memory_variables({}).get("chat_history", [])

    # retrieve previous queries
    previous_queries = [msg.content for msg in chat_history] if chat_history else []
    
    #combine previous queries with the latest one
    query = " ".join(previous_queries[-3:]) + " " + user_input

    response = qa_chain.invoke({"question": query}, config={"callbacks": [langfuse_handler]})
    answer = response["answer"].strip()

    # save query and answer to the memory
    memory.save_context({"question": query}, {"answer": answer})

    # check if the query is too broad
    is_broad, followup_question = check_and_generate_followup(llm_for_chat, chat_history, user_input)
    if is_broad and followup_question:
        return f"{answer}\n\n{followup_question}"

    return answer

def chat_loop(llm_for_chat, qa_chain, memory, langfuse_handler):
    """Start chatting with user using memory and context."""
    st.title("Series Search Assistant")
    st.write("สวัสดีค่ะ! ฉันสามารถช่วยแนะนำซีรีส์หรือให้ข้อมูลเกี่ยวกับนักแสดง ผู้กำกับ และเรื่องย่อของซีรีส์ต่าง ๆ ได้ คุณกำลังมองหาซีรีส์แบบไหนอยู่คะ?")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
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

            # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

            # Get bot response
        response = chatbot_response(llm_for_chat, qa_chain, memory, prompt, langfuse_handler)

            # Display bot response
        with st.chat_message("assistant"):
                st.markdown(response)

            # Add bot response to history
        st.session_state.messages.append({"role": "assistant", "content": response})

        st.rerun()  # Refresh chat UI to maintain conversation history


if __name__ == "__main__":
    langfuse_handler = CallbackHandler()
    df = pd.read_csv('/Users/natrujapatkit/Desktop/4.2/senior project/combined_data.csv')
    texts = df["data"].tolist()
    # Initialize LLM
    llm_for_chat = initialize_llm()

    # Initialize JAI client and embeddings
    client = initialize_jai_client()
    jai_embedding = JAIEmbeddings(client, model_name="jai-emb-passage")

    # Create FAISS vector store
    vector_store = create_vector_store(jai_embedding, texts)

    # Create retriever
    ensemble_retriever = create_retriever(texts, vector_store)

    # Create QA chain
    qa_chain, memory = create_qa_chain(ensemble_retriever, llm_for_chat)

    # Start Streamlit app
    chat_loop(llm_for_chat, qa_chain, memory, langfuse_handler)



