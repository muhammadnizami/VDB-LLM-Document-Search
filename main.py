import PyPDF2
import openai
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import json
import re
from tqdm import tqdm

# Set up your OpenAI API key
openai.api_key = '<OPENAI_API_KEY>'

# Set up QDrant connection
qdrant_client = QdrantClient(
    url="https://992f0808-9ddb-435e-b5cc-2395e73f9b44.ap-northeast-1-0.aws.cloud.qdrant.io:6333",
    api_key="<QDRANT_API_KEY>")
collection_name = "pdf_chunks"

def extract_pdf_chunks(pdf_path, chunk_size, overlap):
    chunks = []
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                chunks.append({'page_num': page_num, 'chunk': chunk})
    return chunks

def get_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",  # Use the ADA model
        input=text,
        max_tokens=1000
    )
    embedding = response['data'][0]['embedding']
    return embedding

def index_chunks_in_vectorDB(chunks):
    # Index the embedding in QDrant vectorDB
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.DOT),
    )
    embeddings = [get_embedding(chunk['chunk']) for chunk in tqdm(chunks)]
    qdrant_client.upsert(
        collection_name=collection_name,
        wait=True,
        points=[PointStruct(
                id=i,
                vector=embeddings[i],
                payload=chunks[i],
            ) for i in range(len(embeddings))])

def search_vectorDB(query_embedding):
    result = qdrant_client.search(
        collection_name=collection_name,
        query_vector = query_embedding,
        limit=10  # You can adjust the limit based on your needs
    )
    return result

def parse_openai_response(openai_response):
    # Split the response into lines and parse each line
    lines = openai_response.strip().split('\n')

    thinking = ""
    has_answer = ""
    answer = ""

    current_var = None
    for line in lines:
        if line.startswith("thinking:"):
            thinking = line.split(':')[1]
            current_var = "thinking"
        elif line.startswith("has_answer:"):
            has_answer = line.split(':')[1]
            current_var = "has_answer"
        elif line.startswith("answer:"):
            answer = line.split(':')[1]
            current_var = "answer"
        elif current_var == "thinking":
            thinking += line
        elif current_var == "has_answer":
            has_answer += line
        elif current_var == "answer":
            answer += line
    
    return thinking, has_answer, answer

def clean_and_lowercase(input_string):
    # Remove non-alphanumeric characters using regular expression
    cleaned_string = re.sub(r'[^a-zA-Z0-9]', '', input_string)
    # Convert to lowercase
    lowercase_string = cleaned_string.lower()
    return lowercase_string

def generate_answer(chunk, question):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": "You are an customer service assistant whose job is to assist the customer service in determining whether a chunk contains answer to a query and generate the appropriate results."},
            {"role": "user", "content": f"Chunk: {chunk}\nQ: {question}\nDoes the chunk contain an answer for the question, either specifically or you can infer the anwer for the question mathematically or logically from an info in the chunk? What should I say to the person who asked this? Give me the answer in three parts: thinking (whether the answer is in the chunk and any mathematical or logical inference if any), answer (only the answer, do not mention anything about the chunk), and has_answer (either yes or no). Give me the results in this format:\nthinking:<thinking>\nanswer:<answer>\nhas_answer:<has_answer>\n"}
        ]
    )
    openai_response = response.choices[0].message.content

    thinking, has_answer, answer = parse_openai_response(openai_response)

    if clean_and_lowercase(has_answer) == "yes":
        return answer
    else:
        return None

def main():
    pdf_path = 'example-document-3.pdf'
    chunk_size = 1000
    overlap = 200
    
    # Extract and index chunks from PDF
    print("reading pdf...")
    chunks = extract_pdf_chunks(pdf_path, chunk_size, overlap)
    print("indexing...")
    index_chunks_in_vectorDB(chunks)
    
    while True:
        user_input = input("Enter your question: ")
        
        # Get user input embedding and search in vectorDB
        user_input_embedding = get_embedding(user_input)
        search_results = search_vectorDB(user_input_embedding)
        
        # Generate answers using GPT-3.5 for each chunk
        answer_found = False
        for result in search_results:
            chunk = chunks[result.id]
            generated_answer = generate_answer(chunk, user_input)
            if (generated_answer):
                print(f"\n----\nPage: {chunk['page_num']}\n----\nPassage:\n{chunk['chunk']}\n----\nPossible Answer: {generated_answer}\nNote: This answer was autogenerated. Please check again in the above passage for more accurate results.")
                print("========================================")
                answer_found = True
        if not answer_found:
            print("Answer not found")

if __name__ == "__main__":
    main()
