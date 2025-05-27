from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, UploadFile
from database import recommendation_collection, rag_collection, rag_names_collection
import torch
from sentence_transformers import SentenceTransformer
import logging
from pydantic import BaseModel
from bson import ObjectId
import datetime
from embeddings_utils import preprocess_text, chunk_pdf, set_model
import os
from dotenv import load_dotenv
import openai
from contextlib import asynccontextmanager

load_dotenv()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device="cpu")
    set_model(model)
    logger.info("Embedding model loaded.")
    yield
    logger.info("Shutting down FastAPI application.")

app = FastAPI(lifespan=lifespan)


def get_embedding(text):
    with torch.no_grad(): # Disable gradient calculation to speed up inference
        return model.encode(text, convert_to_tensor=True).tolist()  # Convert to list for MongoDB compatibility

class Job (BaseModel):
    title: str
    company: str
    location: str
    description: str
    embedding: list = None  # Optional field for embedding, will be set later

@app.post("/recommendation")
async def create_job( request: Request, background_tasks: BackgroundTasks, title : str, company: str, location: str, description: str ):

    recommend = Job(title=title, company=company, location=location, description=description)
    recommend["embedding"] = get_embedding(preprocess_text(recommend["description"]))
    inserted_doc = recommendation_collection.insert_one(recommend)

    return {"message": "Recommendation document created successfully"}

@app.delete("/recommendation/{recommendation_id}")
async def delete_job(recommendation_id: str):
    result = recommendation_collection.delete_one({"_id": ObjectId(recommendation_id)})
    
    if result.deleted_count == 1:
        return {"message": "Recommendation document deleted successfully"}
    else:
        return {"message": "Recommendation document not found"}

@app.patch("/recommendation/{recommendation_id}")
async def update_job(recommendation_id: str, request: Request):
    update_data = await request.json()
    
    if "description" in update_data:
        update_data["embedding"] = get_embedding(update_data["description"])
    
    result = recommendation_collection.update_one(
        {"_id": ObjectId(recommendation_id)},
        {"$set": update_data}
    )
    
    if result.modified_count == 1:
        return {"message": "Recommendation document updated successfully"}
    else:
        return {"message": "Recommendation document not found or no changes made"}

@app.get("/recommendation")
async def get_recommendations(search: str, location: str = None, company: str = None, title: str = None, skip: int = 0, page_size: int = 10):

    query = search.strip()
    if not query:
        return {"message": "Query text is required for recommendations."}

    # filtering and quering
    match_conditions = {}
    regex = {}

    # exact match conditions (before vector search)
    if location := location or request.query_params.get("location"):
            match_conditions["location"] = {"$in": location.split(",")}

    # regex conditions for partial matching (after vector search) (lowers the result count)
    if company := company or request.query_params.get("company"):
        regex["company"] = {"$regex": company, "$options": "i"}
    if title := title or request.query_params.get("title"):
        regex["title"] = {"$regex": title, "$options": "i"}


    embedding = get_embedding(query)


    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector",         # Name of the vector index created in MongoDB

                "path": "embedding",       # Field containing the vector embeddings

                "queryVector": embedding,  # The vector to search for

                "numCandidates": 500,      # Number of candidates to consider (adjust according to collection size)

                "limit": 100,              # Number of results to return (top k results)

                "metric": "cosine",        # Similarity metric to use

                "filter": match_conditions # Filter conditions for the vector search
            }
        },
        # document projection
        {
            "$project": {
                "_id": 1,
                "title": 1,
                'company': 1,
                'location': 1,
                "description": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        },
        # pagination
        {"$skip": skip},
        {"$limit": page_size}
    ]

    # Add regex match conditions if any (after vector search)
    if regex:
        print(regex)
        pipeline.insert(1, {"$match": {"$and": regex}})

    results = list(recommendation_collection.aggregate(pipeline))

    if not results:
        return {"message": "No recommendations found."}

    return results






######################### RAG API Endpoints #########################



@app.post("/rag")
async def create_rag_document(rag: UploadFile, request: Request, background_tasks: BackgroundTasks, chunk_size: int = 500, chunk_overlap: int = 50):

    # check for duplication
    rag_name = rag_names_collection.find_one({"name": file.name.replace(".pdf", "")})
    if rag_name:
        date = rag_name['created_at'].strftime('%Y-%m-%d %I:%M %p GMT')
        return Response({"message": f"Pdf with this name already uploaded on {date}"}, status=status.HTTP_400_BAD_REQUEST)
    
    # make sure temp directory exists
    os.makedirs("./temp", exist_ok=True)
    file_path = None

    try:
        file_path = f"./temp/{rag.filename}"
        with open(file_path, "wb") as f:
            f.write(pdf.file.read())
        
        chunk_size = request.get("chunk_size", 500)
        chunk_overlap = request.get("chunk_overlap", 50)
        pdf_name = pdf.filename if pdf else request.get("name")
        chunks = chunk_pdf(pdf_name = pdf_name, file_path = file_path )
        
        rag_names_collection.insert_one({"name": pdf.filename.replace(".pdf", "") if pdf else request.get("name"), 'url': request.get("url") if request.get("url") is not None else "", "created_at": datetime.utcnow()})
        rag_collection.insert_many(chunks)
        
        if file_path is not None: 
            os.remove(file_path)

        return {"message": f"PDF uploaded and processed successfully"}
    except Exception as e:
        if file_path is not None: 
            os.remove(file_path)
        print(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail="PDF processing failed")

@app.get("/rag")
async def get_rag_documents(request: Request):
    return list(rag_names_collection.find())

@app.delete("/rag")
async def delete_rag_documents(name: str = None, rag_id: str = None):
    if rag_id:
        name = rag_names_collection.find_one({"_id": ObjectId(id)})['name']
        if name:
            result = rag_collection.delete_many({"metadata": name})
            rag_names_collection.delete_one({"_id": ObjectId(id)})
    elif name:
        result = rag_collection.delete_many({"metadata": name})
    else:
        raise HTTPException(status_code=500, detail="No name or id provided")

    if result.deleted_count > 0:
        return {"message": "RAG document deleted successfully, deleted chunks count: " + str(result.deleted_count)}
    else:
        return {"message": "RAG document not found"}
    
@app.get("/rag_question")
async def get_rag_questions(request: Request, question: str):
    """
    Performs a RAG query against indexed PDF data in MongoDB, includes chat history,
    and asks OpenAI.
    """
    if len(question) < 1:
        raise HTTPException(status_code=500, detail="No question provided")
    query_text = question
    print(f"Received query: '{query_text}'")

    try:
        # 1. Get embedding for the user query
        query_embedding = get_embedding(query_text)

        if not query_embedding:
             raise HTTPException(status_code=500, detail="Failed to generate embedding for the query.")

        # 2. Search MongoDB for relevant chunks using vector search
        search_results = list(rag_collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "rag_index",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 1000,
                    "limit": 10,
                    "metric": "cosine"
                }
            },
            {
                 "$project": {
                    "_id": 0,
                    "text": 1,
                    "metadata": 1,
                    "score": { "$meta": "vectorSearchScore" }
                 }
            }
        ]))

        # 3. Format the retrieved chunks as context
        context = "\n\n---\n\n".join([doc["text"] for doc in search_results])

        # Handle case where PDF doesn't exist or no relevant chunks found for current query
        if not search_results:
            print("No new relevant chunks found, relying on prompt.")


        # 4. Prepare messages for OpenAI, including history and context
        # keep modifing the prompt to get better results
        messages = [
            {
                "role": "system",
                "content": (
                    "Dont metion anything about being looking at a provided context. "
                    "You are a helpful assistant that answers questions based on the provided context. "
                    "You can use external knowledge if needed but with a focus on the provided context. "
                )
            }
        ]

        # Add the current user query, incorporating the retrieved context
        current_user_message_content = f"Context:\n{context}\n\nQuestion: {query_text}"

        messages.append({
            "role": "user",
            "content": current_user_message_content
        })

        print(f"Sending {len(messages)} messages to OpenAI API.")

        # 5. Call OpenAI API
        key = os.getenv('OPEN_AI')
        if not key:
            raise HTTPException(status_code=500, detail="OpenAI API key not found.")
        openai.api_key = key
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo", # change according to api quota
            messages=messages,
            temperature=0.1, # Higher temperature means more random responses, lower temperature means more focused responses
        )

        # 6. Extract the answer
        answer = response.choices[0].message.content.strip()
        print("Answer:", answer)
        return {"answer": answer}

    except HTTPException as e:
        # Re-raise HTTPExceptions
        raise e
    except Exception as e:
        print(f"An error occurred during RAG query with history: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")