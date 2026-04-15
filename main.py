from fastapi import FastAPI 
from src.api.v1.routes.query_routes import router as query_router
from src.api.v1.routes.upload_routes import router as upload_router

#create a FastAPI instance
app = FastAPI(title = "MULTIMODAL RAG SYSTEM") 

# we will enable rest api endpoint at localhost:8000/
@app.get("/")
def read_root():
    return {
        "message": "Hello World!"
    }

#health check endpoint (to find whether this is working or not)
@app.get("/health")
def health_check():
    return{
        "status":"ok"
    }

app.include_router(query_router,prefix="/api/v1")
app.include_router(upload_router, prefix="/api/v1")