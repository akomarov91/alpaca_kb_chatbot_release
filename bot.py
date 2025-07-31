#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pickle
import openai
import time
import os
import streamlit as st

from chat_logger_with_weekly_rotation import log_chat_to_csv

api_key = st.secrets['openai']['api_key']

client = openai.OpenAI(api_key=api_key)

# Function for embedding
def get_embedding(text, model="text-embedding-3-small"):
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding


# Check if embeddings exist
EMBEDDINGS_PATH = "embeddings.pkl"

if os.path.exists(EMBEDDINGS_PATH):
    print("Loading precomputed embeddings...")
    with open(EMBEDDINGS_PATH, "rb") as f:
        data = pickle.load(f)
        chunks = data["chunks"]
        chunk_embeddings = data["embeddings"]
else:
    print("Embedding chunks and saving to disk...")
    
    # open doc file from Alpaca
    from docx import Document
    doc = Document("Alpaca Docs-v13-w-links.docx")

    # Extract text from all paragraphs
    full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip() != ""])


    # Split into chunks for embeddings
    def chunk_text(text, max_chunk_size=1000, overlap=100):
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += max_chunk_size - overlap
        return chunks


    chunks = chunk_text(full_text, max_chunk_size=1000, overlap=100)
    # Get chunks for embedding
    def chunk_dialogues_from_docx(text):
        blocks = text.split("User:")
        return [block.strip() for block in blocks if block.strip()]

    #chunks = chunk_dialogues_from_docx(full_text)

    chunk_embeddings = []
    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i + 1} of {len(chunks)}", end="\r")
        chunk_embeddings.append(get_embedding(chunk))
        time.sleep(0.2)  # to avoid rate limiting

    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": chunk_embeddings}, f)
    print("\nSaved embeddings to embeddings.pkl")


# Build Nearest Neighbors Index
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=5, metric='cosine')
nn.fit(chunk_embeddings)

# Define Retrieval Function
def retrieve_relevant_chunks(query, k=5):
    query_embedding = get_embedding(query)
    distances, indices = nn.kneighbors([query_embedding], n_neighbors=k)
    return [chunks[i] for i in indices[0]]


# Define Chat Function Grounded to Alpaca
def chat_with_alpaca(query):
    relevant_chunks = retrieve_relevant_chunks(query)
    context = "\n---\n".join(relevant_chunks)

    prompt = f"""
You are a helpful assistant. Use only the provided Alpaca QA excerpts to answer the user's question.
You may make simple logical inferences if they directly follow from the information in the text.
Here are relevant passages:

{context}

Now respond to this query. If the answer cannot be directly found or inferred from the text, say: "I can't find the answer to your question in the Alpaca knowledge base, please contact support at: https://hd.service.alpacamed.com/"

User: {query}
Answer:
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    # Step 4: Get and clean the response
    answer = response.choices[0].message.content.strip()

    #  Step 5: Log the user question and bot answer
    log_chat_to_csv(query, answer)

    # Step 6: Return the response
    return answer



