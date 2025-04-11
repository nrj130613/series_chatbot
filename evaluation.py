from openai import OpenAI
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

def initialize_openai():
    """Load OpenAI API key from environment variables."""
    load_dotenv()  # Load environment variables from a .env file (if applicable)
    api_key = os.getenv("OPENAI_API_KEY")  # Ensure you have set this variable
    if not api_key:
        raise ValueError("Missing OpenAI API Key. Set 'OPENAI_API_KEY' in your environment variables.")
    return api_key

# Initialize OpenAI API
openai_api_key = initialize_openai()


def response_eval_with_llm(query, reference_answer, generated_response):
    """Evaluates the LLM-generated response against the reference answer."""
    llm_as_judge = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=500, openai_api_key=openai_api_key)
    llm_judge_result = llm_as_judge.invoke(
        f"""
        You are an AI evaluator for a Retrieval-Augmented Generation (RAG) system. 
        Your task is to judge the quality of a generated response by comparing it to a reference answer, 
        specifically in the context of products and their related information.

        ### **Evaluation Guidelines:**
        - **Relevance**: Judge whether the generated response directly addresses the query and contains information that aligns with the reference answer.
        - **Usefulness**: Evaluate if the response provides valuable, insightful, or helpful information—whether it adds clarity, depth, or actionable details.
        - **Truthfulness**: Ensure the response is factually accurate, free of misinformation, and does not misrepresent details.
        
        - Variations in wording are acceptable as long as the response covers the key information.
        - A response should not be penalized for different phrasing as long as it is factually correct, useful, and relevant.

        ### **Scoring Scale (1-5):**
        - **5 = Excellent** (Fully relevant, highly useful, and factually accurate.)
        - **4 = Good** (Mostly relevant, useful, and truthful, with minor gaps.)
        - **3 = Adequate** (Somewhat relevant or useful but with noticeable omissions or minor inaccuracies.)
        - **2 = Poor** (Barely relevant, not very useful, or contains some incorrect information.)
        - **1 = Unacceptable** (Off-topic, misleading, or factually incorrect.)

        Provide only a JSON response in the following format:
        {{
            "response_relevance_score": <score>,
            "response_usefulness_score": <score>,
            "response_truthfulness_score": <score>,
            "response_justification": "<explanation>"
        }}
        
        ---

        **Query:** {query}
        **Reference Answer:** {reference_answer}
        **Generated Response:** {generated_response}
        """
    )
    
    return llm_judge_result.content

def judge_correctness_with_llm(query, reference_documents, retrieved_documents):
    """Evaluates the relevance of retrieved documents to the query."""
    llm_as_judge = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=500, openai_api_key=openai_api_key)
    llm_judge_result = llm_as_judge.invoke(
        f"""
        You are an AI evaluator for a Retrieval-Augmented Generation (RAG) system. 
        Your task is to judge the relevance of retrieved documents to a given query, 
        specifically in the context of products and their related information.

        ### **Evaluation Guidelines:**
        - Judge whether the retrieved documents mention any product or keywords related to the query.
        - A document should be considered relevant if it contains any of the key product names, product description, or related keywords from the query.
        - Additionally, compare the retrieved documents with reference (gold standard) documents to ensure relevance alignment.

        ### **Scoring Scale:**
        - **5 = Highly relevant** (The retrieved documents contain direct mentions of the queried products or clear keyword matches and align well with the reference documents.)
        - **4 = Mostly relevant** (The documents mention related products or categories but may lack some details; alignment with reference documents is strong but not perfect.)
        - **3 = Somewhat relevant** (The documents mention similar or related products but not the exact ones in the query; partial alignment with reference documents.)
        - **2 = Barely relevant** (The documents are loosely related to the topic but do not focus on the requested products; minimal alignment with reference documents.)
        - **1 = Not relevant** (The documents are about completely different products or unrelated topics; do not align with reference documents.)

        Provide only a JSON response in the following format:
        {{
            "doc_relevance_score": <score>,
            "doc_justification": "<explanation>"
        }}
        
        ---

        **Query**: {query}
        **Reference Documents**: {reference_documents}
        **Retrieved Documents**: {retrieved_documents}
        """
    )
    return llm_judge_result.content

query = "Is product XYZ safe for pregnant women?"
reference_docs = ["Product XYZ contains folic acid, which is beneficial during pregnancy."]
retrieved_docs = ["Product XYZ has high vitamin A, which should be limited in pregnancy."]

# Run Evaluation
result = judge_correctness_with_llm(query, reference_docs, retrieved_docs)
print(result)