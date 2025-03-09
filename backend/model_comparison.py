from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
from models import SentimentService
from database import get_db, Analysis
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

app = FastAPI()
sentiment_service = SentimentService()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = "your-very-secure-secret-key-here"
ALGORITHM = "HS256"

class ModelComparisonResponse(BaseModel):
    model: str
    sentiment: str
    confidence: float

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

@app.post("/api/model-comparison", response_model=List[ModelComparisonResponse])
async def compare_models(
    text: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    results = []
    for model_name in sentiment_service.models.keys():
        sentiment, confidence = sentiment_service.analyze(text, model_name)
        analysis = Analysis(
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            model_used=model_name,
            user_id=current_user.id
        )
        db.add(analysis)
        results.append({
            "model": model_name,
            "sentiment": sentiment,
            "confidence": confidence
        })
    db.commit()
    return results
