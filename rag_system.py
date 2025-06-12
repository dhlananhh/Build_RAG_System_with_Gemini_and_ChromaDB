import requests
from pypdf import PdfReader
import os
import re
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
from typing import List
from dotenv import load_dotenv

load_dotenv()

def download_pdf(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading PDF from {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"PDF downloaded successfully to {save_path}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading PDF: {e}")
            exit()
    else:
        print(f"PDF file already exists at {save_path}. Skipping download.")

pdf_url = "https://services.google.com/fh/files/misc/ai_adoption_framework_whitepaper.pdf"
pdf_path = "ai_adoption_framework_whitepaper.pdf"
download_pdf(pdf_url, pdf_path)

def load_pdf(file_path):
    if not os.path.exists(file_path):
        print(f"Error: PDF file not found at {file_path}")
        return ""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {e}")
        return ""
pdf_text = load_pdf(pdf_path)
if not pdf_text:
    print("Failed to load PDF text. Exiting.")
    exit()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found. Please ensure it's set in your .env file or system environment variables.")
try:
    genai.configure(api_key=gemini_api_key)
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"Failed to configure Gemini API: {e}")
    exit()

def split_text(text: str) -> List[str]:
    chunks = re.split(r'\n\s*\n', text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]
chunked_text = split_text(pdf_text)
if not chunked_text:
    print("Failed to chunk the text.")
    exit()

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = "models/embedding-001"
        title = "PDF Document Segments"
        try:
            response = genai.embed_content(
                model=model,
                content=input,
                task_type="retrieval_document",
                title=title
            )
            return response["embedding"]
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return [] * len(input)

db_folder = "chroma_db"
if not os.path.exists(db_folder):
    try:
        os.makedirs(db_folder)
        print(f"Created database directory: {db_folder}")
    except OSError as e:
        print(f"Error creating directory {db_folder}: {e}")
        exit()

db_name = "rag_experiment_v2"
db_path = os.path.join(os.getcwd(), db_folder)
chroma_client = chromadb.PersistentClient(path=db_path)

try:
    db = chroma_client.get_collection(name=db_name, embedding_function=GeminiEmbeddingFunction())
    print(f"Loaded existing Chroma collection: {db_name}")
except Exception:
    print(f"Collection {db_name} not found. Creating new collection...")
    try:
        db = chroma_client.create_collection(name=db_name, embedding_function=GeminiEmbeddingFunction())
        print("Adding documents to the new collection...")
        ids = [str(i) for i in range(len(chunked_text))]
        db.add(documents=chunked_text, ids=ids)
        print(f"Created and populated Chroma collection: {db_name}")
    except Exception as e:
        print(f"Error creating or populating Chroma collection: {e}")
        exit()

def get_relevant_passages(query: str, db, n_results: int) -> List[str]:
    try:
        results = db.query(query_texts=[query], n_results=n_results)
        if results and 'documents' in results and results['documents'] and results['documents'][0]:
            return results['documents'][0]
        else:
            print("Warning: No relevant documents found or unexpected query result format.")
            return []
    except Exception as e:
        print(f"Error querying Chroma database: {e}")
        return []

def make_rag_prompt(query: str, relevant_passages: List[str]) -> str:
    combined_passage = "\n---\n".join(relevant_passages)
    escaped_passage = combined_passage.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")
    prompt = f"""
        You are a helpful and informative bot that answers questions using text from the reference passage(s) included below.
        Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.
        However, you are talking to a non-technical audience, so be sure to break down complicated concepts and
        strike a friendly and conversational tone. Use only the information from the PASSAGE(s) to answer the question.

        QUESTION: '{query}'

        PASSAGE(S):
        '{escaped_passage}'

        ANSWER:
    """
    return prompt

def generate_answer(prompt: str) -> str:
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating answer with Gemini: {e}")
        return "Sorry, I couldn't generate an answer due to an error."

def run_interactive_qa():
    print("\n--- RAG Query Interface ---")
    print("Type 'exit' or 'quit' to stop.")
    while True:
        query = input("Please enter your query: ")
        if query.lower() in ['exit', 'quit']:
            print("Exiting.")
            break
        if not query:
            print("No query provided. Please try again.")
            continue
        print("Searching for relevant information...")
        relevant_passages = get_relevant_passages(query, db, n_results=3)
        if not relevant_passages:
            print("No relevant information found for the given query in the document.")
            continue
        print("Generating answer...")
        final_prompt = make_rag_prompt(query, relevant_passages)
        answer = generate_answer(final_prompt)
        print("\nGenerated Answer:")
        print(answer)
        print("-" * 20)

if __name__ == "__main__":
    initial_query = "What is the AI Maturity Scale?"
    print(f"Running initial query: '{initial_query}'")
    relevant_passages = get_relevant_passages(initial_query, db, n_results=3)
    if relevant_passages:
        final_prompt = make_rag_prompt(initial_query, relevant_passages)
        answer = generate_answer(final_prompt)
        print("\nInitial Query Answer:")
        print(answer)
    else:
        print("Could not find relevant info for the initial query.")

    run_interactive_qa()
