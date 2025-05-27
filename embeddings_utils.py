from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import fitz  # PyMuPDf
from io import BytesIO
import torch
from fastapi import HTTPException

model = None  # Will be set from outside
def set_model(m):
    global model
    model = m

def get_embedding(text):
    if model is None:
        raise ValueError("Model is not initialized. Use `set_model()` first.")
    with torch.no_grad():
        return model.encode(text, convert_to_tensor=True).tolist()

def preprocess_text(text):
    text = text.replace('\n', ' ').strip() # Remove newlines and leading/trailing spaces
    text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with a single space
    return text.lower() # Convert to lowercase

def chunk_pdf(pdf_name: str, file_path: str = None, chunk_size: int = 500, chunk_overlap: int = 50):
    print(f"Processing PDF with PyPDFLoader: {file_path}")
    try:
        # Load PDF using PyMuPDF
        if file_path is not None:
            pdf_stream = BytesIO(open(file_path, "rb").read())
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            text = "\n".join([page.get_text("text") for page in doc])
            doc.close()

        if not text:
            raise HTTPException(status_code=400, detail="No text found in PDF.")
            
        text = preprocess_text(text)

        # Split the text into chunks and embed
        #- could use fastapi background tasks to make this in the background -#
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,   # Number of characters per chunk
            chunk_overlap=50, # Number of characters to overlap between chunks
            length_function=len,
        )
        chunked_text = text_splitter.split_text(text)
        
        chunks_with_metadata = []
        for chunk in chunked_text:
            chunk_data = {
                "text": chunk,
                "embedding": get_embedding(chunk),
                "metadata": pdf_name # for relevance, and delete later
            }
            chunks_with_metadata.append(chunk_data)
        
        print(f"Successfully split PDF into {len(chunks_with_metadata)} chunks.")
        return chunks_with_metadata
    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")
        # Raise HTTPException to return a proper API error response
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")
