from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()



#- MongoDB Atlas Connection  -#

MONGO_URI = os.getenv("MONGO_URI") # Mongo Atlas Connection String

client = MongoClient(MONGO_URI) # Connect to MongoDB Atlas



#- Database and Collection names -#

DB_NAME = "vector_db" # Mongo Atlas database name

db = client[DB_NAME] # Connect to the database

rag_collection = db["rag"] # Collection for RAG documents
rag_names_collection = db["rag_name"] # Collection for RAG names (prevent duplicates and manage deletions)
recommendation_collection = db["recommendation"] # Collection for recommendation