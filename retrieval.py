from uuid import uuid4
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.chains import HypotheticalDocumentEmbedder
from langchain_community.vectorstores import FAISS
from openai import OpenAI
import faiss


# HYDE Function (Modified for JAI)
def create_hyde(embedding_model):
    """Create HYDE-based embedding using JAI"""
    llm_for_search = OpenAI()  
    embeddings = HypotheticalDocumentEmbedder.from_llm(
        llm_for_search, 
        embedding_model,  
        prompt_key="web_search"
    )
    return embedding_model, embeddings


def create_vector_store(embedding_model, texts):
    """Create FAISS vector store with JAI embeddings and add data"""
    document_embeddings = embedding_model.embed_documents(texts)
    index = faiss.IndexFlatL2(len(document_embeddings[0]))  # Create FAISS index

    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    uuids = [str(uuid4()) for _ in range(len(texts))]
    vector_store.add_texts(texts=texts, ids=uuids)
    
    return vector_store


def create_retriever(texts, vector_store):
    """create Retriever using FAISS and BM25"""
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    bm25_retriever = BM25Retriever.from_texts(texts)
    bm25_retriever.k = 3

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever