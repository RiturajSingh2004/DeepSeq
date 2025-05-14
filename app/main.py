from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.api.routes import embedding, properties, classification, stability, structure

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Protein Analysis Platform API",
    description="API for AI-driven protein sequence analysis with embeddings, property predictions, and structure analysis",
    version="1.0.0",
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(embedding.router, prefix="/api/embedding", tags=["Embedding"])
app.include_router(properties.router, prefix="/api/properties", tags=["Properties"])
app.include_router(classification.router, prefix="/api/classification", tags=["Classification"])
app.include_router(stability.router, prefix="/api/stability", tags=["Stability"])
app.include_router(structure.router, prefix="/api/structure", tags=["Structure"])

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Protein Analysis Platform API",
        "docs": "/docs",
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)