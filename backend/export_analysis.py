from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from database import get_db, Analysis, User
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
import csv
from io import StringIO

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = "your-very-secure-secret-key-here"
ALGORITHM = "HS256"

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

@app.get("/api/export-analyses")
async def export_analyses(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    analyses = db.query(Analysis).filter(Analysis.user_id == current_user.id).all()
    if not analyses:
        raise HTTPException(status_code=404, detail="No analyses found to export")

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Text", "Sentiment", "Confidence", "Model Used", "Created At"])
    for analysis in analyses:
        writer.writerow([
            analysis.id,
            analysis.text,
            analysis.sentiment,
            analysis.confidence,
            analysis.model_used,
            analysis.created_at.isoformat()
        ])
    
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment;filename=analysis_history.csv"}
    )
