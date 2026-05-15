import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import BaseModel, Field
from typing import List
import threading

def execute_query(graph_obj, query, result_list, index):
    """Funzione helper per eseguire una query e salvare il risultato."""
    try:
        records = graph_obj.query(query)
        result_list[index] = records
    except Exception as e:
        print(f"Errore durante l'esecuzione della query: {e}")
        result_list[index] = None # O un indicatore di errore specifico

def structured_retriever(graph, question: str, entity_chain) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    try:
        generated_cypher = entity_chain.invoke(
            {
                "question": question,
                "schema": graph.schema,
            }
        )

        if generated_cypher:
            generated_cypher = json.loads(generated_cypher)

            main_query = generated_cypher["main_query"]
            document_distinct_query = generated_cypher["document_distinct_query"]
            
            results = [None, None] # results[0] per main_query, results[1] per document_distinct_query

            thread1 = threading.Thread(target=execute_query, args=(graph, main_query, results, 0))
            thread2 = threading.Thread(target=execute_query, args=(graph, document_distinct_query, results, 1))

            # Avvia i thread
            thread1.start()
            thread2.start()

            # Attendi che entrambi i thread abbiano completato l'esecuzione
            thread1.join()
            thread2.join()

            records = results[0]
            documents = results[1]

            return records, documents

        return None, []
    
    except Exception as ex:
        print(ex)
        return None,[]

def grade_document(question: str, documents):

    llm = AzureChatOpenAI(
        azure_deployment="gpt-5.4",
        openai_api_version="2024-12-01-preview",
        temperature=0.7 #more creative and less deterministic responses
    )

    relevance_prompt = """
You are an expert in information retrieval and semantic analysis.

## Goal
You will be provided with a user query and a list of document chunks. Your task is to carefully analyze the meaning and intent of the user query, and select only the chunks that are truly relevant to answering or addressing the query.

## Instructions

- Read and understand the user query in detail.
- Carefully analyze each provided chunk (title, content, and keywords).
- For each chunk, decide if it is genuinely relevant and useful for the query—select only chunks that directly answer, match, or provide substantial information for the user's needs.
- Do not include chunks that are only loosely related, contain background information, or do not contribute meaningfully to the user query.
- You must be strict in your selection: only return chunks that you would judge as clearly relevant by the standards of an expert information specialist.
- Maintain the original order of the chunks that you select.
- Do not invent, rewrite, or summarize the content of the chunks. Return only the chunks as provided.

## Output Format

Return your response in the following JSON format:

{
  "relevant_documents": ["DOCUMENT_CONTENT1", "DOCUMENT_CONTENT2", ...]
}

- The output must be only valid JSON, with no extra explanation or commentary.
"""

    class Documents(BaseModel):
        relevant_documents: List[str] = Field(
            description="The content of the selected document"
        )


    result = llm.with_structured_output(Documents).invoke(
        [
            SystemMessage(content=relevance_prompt),
            HumanMessage(content=f"Query:{question}\nDocuments to analyze: {documents}")
        ] 
    )

    return result.relevant_documents

def get_stractered_data(graph, question: str, entity_chain):
    structured_data, documents = structured_retriever(graph, question, entity_chain)

    return structured_data, documents

def get_unstructured_data(vector_index, question: str):
    documents = vector_index.similarity_search_with_score(question)

    unstructured_data = []
    for document in documents:
        unstructured_data.append(document[0].page_content)
    
    return unstructured_data

def grade_unstractered_data(question: str, unstructured_data: List[str]) -> str:
    return grade_document(question=question, documents=unstructured_data)

def get_context(structured_data,documents):
    final_data = f"""Structured data:\n
    {structured_data}\n
    Unstructured data:\n
    {"#Document ". join(documents)}
    """
    return final_data
        