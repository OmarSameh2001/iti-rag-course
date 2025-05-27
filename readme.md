# Application setup:
Install the requirements.txt in python venv

make .env file with API keys and connection strings

run using: `uvicorn app:main --reload`

# Prerequisites:
OPEN-AI API key or local infered LLM

setup vector indexes in mongo db

# Toturials:
## Vector Database:
https://www.youtube.com/watch?v=gl1r1XV0SLw
## Mongo setup:
https://www.youtube.com/watch?v=yMdEsZOBJhI
### My index vector setup
`
{
  "mappings": {
    "dynamic": true,
    "fields":
      "=embedding": {
        "dimensions": 384,
        "similarity": "cosine",
        "type": "knnVector"
      }
  }
}
`
