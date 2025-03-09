from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from typing import Dict
from pydantic import BaseModel
from database import get_db, Analysis, User
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = "your-very-secure-secret-key-here"
ALGORITHM = "HS256"

class TextStatsResponse(BaseModel):
    total_analyses: int
    avg_word_count: float
    sentiment_distribution: Dict[str, int]

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

@app.get("/api/text-statistics", response_model=TextStatsResponse)
async def get_text_stats(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    total_analyses = db.query(Analysis).filter(Analysis.user_id == current_user.id).count()
    analyses = db.query(Analysis).filter(Analysis.user_id == current_user.id).all()
    word_counts = [len(text.text.split()) for text in analyses]
    avg_word_count = sum(word_counts) / len(word_counts) if word_counts else 0
    sentiment_counts = db.query(Analysis.sentiment, func.count(Analysis.id))\
        .filter(Analysis.user_id == current_user.id)\
        .group_by(Analysis.sentiment).all()
    sentiment_distribution = {sentiment: count for sentiment, count in sentiment_counts}
    return {
        "total_analyses": total_analyses,
        "avg_word_count": avg_word_count,
        "sentiment_distribution": sentiment_distribution
    }
