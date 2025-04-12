import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain.embeddings.base import Embeddings

# Initialize JAI Client
def initialize_jai_client():
    return OpenAI(api_key=st.secrets['JAI_API_KEY'], base_url=st.secrets['CHAT_BASE_URL'])

# JAI Embedding Class
class JAIEmbeddings(Embeddings):  
    def __init__(self, client, model_name="jai-emb-passage"):
        self.client = client
        self.model_name = model_name  

    def embed_documents(self, texts):
        response = self.client.embeddings.create(input=texts, model="jai-emb-passage")
        return [res.embedding for res in response.data]

    def embed_query(self, query):
        response = self.client.embeddings.create(input=query, model="jai-emb-query")
        return response.data[0].embedding  

    def embed(self, texts):
        return self.embed_documents(texts)