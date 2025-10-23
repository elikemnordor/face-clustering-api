import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from insightface.app import FaceAnalysis
from PIL import Image
import uvicorn

# ---------------------------------------
# ✅ App & Database Setup
# ---------------------------------------
app = FastAPI(title="Face Clustering API")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ---------------------------------------
# ✅ Database Model
# ---------------------------------------
class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    face_id = Column(String, index=True)
    embedding = Column(LargeBinary)
    image_url = Column(String)
    similarity = Column(Float)

Base.metadata.create_all(bind=engine)

# ---------------------------------------
# ✅ Initialize Face Model
# ---------------------------------------
app_insight = FaceAnalysis(name="buffalo_l")
app_insight.prepare(ctx_id=0, det_size=(640, 640))

# ---------------------------------------
# ✅ Utilities
# ---------------------------------------
def get_embedding(image: Image.Image):
    arr = np.array(image)
    faces = app_insight.get(arr)
    if not faces:
        return None
    return faces[0].embedding.astype(np.float32)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------------------------------------
# ✅ Routes
# ---------------------------------------

@app.get("/")
def home():
    return {"message": "Face Clustering API is running"}

@app.post("/process-face")
async def process_face(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        embedding = get_embedding(image)

        if embedding is None:
            return JSONResponse(content={"error": "No face detected"}, status_code=400)

        session = SessionLocal()
        embeddings = session.query(FaceEmbedding).all()

        # Compare to existing embeddings
        best_match = None
        best_score = 0.0
        for e in embeddings:
            existing_emb = np.frombuffer(e.embedding, dtype=np.float32)
            sim = cosine_similarity(embedding, existing_emb)
            if sim > best_score:
                best_score = sim
                best_match = e

        # Threshold for new face
        THRESHOLD = 0.6
        if best_match and best_score >= THRESHOLD:
            face_id = best_match.face_id
        else:
            face_id = f"face_{len(embeddings)+1}"

        # Save new embedding
        new_entry = FaceEmbedding(
            face_id=face_id,
            embedding=embedding.tobytes(),
            image_url=file.filename,
            similarity=best_score,
        )
        session.add(new_entry)
        session.commit()
        session.close()

        return {
            "face_id": face_id,
            "similarity": float(best_score),
            "message": "Processed successfully",
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ---------------------------------------
# ✅ Start Server (for local + Railway)
# ---------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
