from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
from pathlib import Path

# Load heart disease dataset
dataset_path = Path("Datasets/heart.csv")
df = pd.read_csv(dataset_path)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_heart_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        # Create feature string excluding target
        features = ", ".join([f"{col}: {row[col]}" for col in df.columns if col != 'target'])
        
        document = Document(
            page_content=features,
            metadata={
                "target": row["target"],
                "id": str(i)
            }
        )
        ids.append(str(i))
        documents.append(document)
        
vector_store = Chroma(
    collection_name="heart_disease",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(search_kwargs={"k": 5})