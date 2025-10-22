from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Float, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import numpy as np
import insightface
import io
from PIL import Image
import json
import os

app = FastAPI(title="Face Embedding API")

# -----------------------------
# DATABASE SETUP
# -----------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    face_id = Column(String, unique=False)
    embedding = Column(JSON)  # store as list
    avg_similarity = Column(Float, default=0.0)

Base.metadata.create_all(bind=engine)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=0, det_size=(640, 640))

# -----------------------------
# HELPERS
# -----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    faces = model.get(np.array(img))
    if not faces:
        return None
    return faces[0].embedding.tolist()

# -----------------------------
# API ENDPOINT
# -----------------------------
@app.post("/process-face")
async def process_face(file: UploadFile = File(...)):
    db = SessionLocal()
    content = await file.read()
    new_embedding = get_embedding(content)
    if new_embedding is None:
        return {"error": "no face detected"}

    new_vec = np.array(new_embedding)
    faces = db.query(FaceEmbedding).all()

    best_match_id = None
    best_score = 0.0

    for f in faces:
        old_vec = np.array(f.embedding)
        score = cosine_similarity(new_vec, old_vec)
        if score > best_score:
            best_score = score
            best_match_id = f.face_id

    threshold = 0.6  # tweak for stricter/looser matching

    if best_score < threshold or not best_match_id:
        # new face
        new_face_id = f"face_{len(faces)+1}"
        db.add(FaceEmbedding(face_id=new_face_id, embedding=new_embedding))
        db.commit()
        db.close()
        return {"face_id": new_face_id, "match": False, "similarity": best_score}
    else:
        db.close()
        return {"face_id": best_match_id, "match": True, "similarity": best_score}
