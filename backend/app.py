import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Depends, HTTPException, Form, Request, status, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func
from transformers import pipeline
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
import csv
from io import StringIO
import json
import uvicorn
import re
import shutil
import tempfile
from starlette.middleware.cors import CORSMiddleware

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("sentiment_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Application Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sentiment.db")
SECRET_KEY = os.getenv("SECRET_KEY", "your-very-secure-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
ALLOWED_MODELS = ["default", "distilbert", "roberta"]
UPLOAD_DIR = "uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Advanced Sentiment Analysis Tool",
    description="A comprehensive tool for text sentiment analysis with user management and batch processing.",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Templates and Static Files ---
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Ensure Upload Directory Exists ---
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# --- Database Setup ---
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Database Models ---
class User(Base):
    """User model for storing user information."""
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=True)
    full_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=func.now())
    last_login = Column(DateTime, nullable=True)
    analyses = relationship("Analysis", back_populates="user")
    preferences = relationship("UserPreference", uselist=False, back_populates="user")

class UserPreference(Base):
    """User preferences for customization."""
    __tablename__ = "user_preferences"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    default_model = Column(String, default="default")
    theme = Column(String, default="light")
    notifications_enabled = Column(String, default="yes")
    user = relationship("User", back_populates="preferences")

class Analysis(Base):
    """Analysis model for storing sentiment analysis results."""
    __tablename__ = "analyses"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    sentiment = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    model_used = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=func.now())
    user = relationship("User", back_populates="analyses")

Base.metadata.create_all(bind=engine)

# --- Pydantic Models ---
class Token(BaseModel):
    access_token: str
    token_type: str
    refresh_token: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = Field(None, min_length=8)

class UserPreferenceUpdate(BaseModel):
    default_model: Optional[str] = None
    theme: Optional[str] = None
    notifications_enabled: Optional[str] = None

class AnalysisResponse(BaseModel):
    id: int
    text: str
    sentiment: str
    confidence: float
    model_used: str
    created_at: datetime

class BatchAnalysisRequest(BaseModel):
    texts: List[str]
    model: str = "default"

# --- Database Dependency ---
def get_db():
    """Provide a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Authentication Setup ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password for storage."""
    return pwd_context.hash(password)

def get_user(db: Session, username: str) -> Optional[User]:
    """Retrieve a user by username."""
    return db.query(User).filter(User.username == username).first()

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Authenticate a user with username and password."""
    user = get_user(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    user.last_login = datetime.utcnow()
    db.commit()
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict) -> str:
    """Create a JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """Get the current authenticated user from the token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            raise credentials_exception
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(db, username)
    if user is None:
        raise credentials_exception
    return user

# --- Sentiment Analysis Service ---
class SentimentService:
    """Service for handling sentiment analysis with multiple models."""
    def __init__(self):
        self.models = {
            "default": pipeline("sentiment-analysis"),
            "distilbert": pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"),
            "roberta": pipeline("sentiment-analysis", model="roberta-base"),
        }
        logger.info("Sentiment models initialized.")

    def analyze(self, text: str, model_name: str = "default") -> tuple[str, float]:
        """Analyze the sentiment of a given text."""
        if not text.strip():
            raise ValueError("Text cannot be empty.")
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available. Choose from {list(self.models.keys())}")
        try:
            result = self.models[model_name](text)[0]
            return result["label"], result["score"]
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Analysis failed due to internal error.")

sentiment_service = SentimentService()

# --- Helper Functions ---
def create_user(db: Session, user: UserCreate) -> User:
    """Create a new user in the database."""
    if not re.match(r"^[a-zA-Z0-9_]+$", user.username):
        raise HTTPException(status_code=400, detail="Username can only contain letters, numbers, and underscores.")
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, hashed_password=hashed_password, email=user.email, full_name=user.full_name)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    # Create default preferences
    db_prefs = UserPreference(user_id=db_user.id)
    db.add(db_prefs)
    db.commit()
    logger.info(f"User '{user.username}' created successfully.")
    return db_user

def update_user(db: Session, username: str, user_update: UserUpdate) -> User:
    """Update an existing user's information."""
    db_user = get_user(db, username)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    if user_update.email:
        db_user.email = user_update.email
    if user_update.full_name:
        db_user.full_name = user_update.full_name
    if user_update.password:
        db_user.hashed_password = get_password_hash(user_update.password)
    db.commit()
    db.refresh(db_user)
    logger.info(f"User '{username}' updated successfully.")
    return db_user

def delete_user(db: Session, username: str) -> None:
    """Delete a user from the database."""
    db_user = get_user(db, username)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(db_user)
    db.commit()
    logger.info(f"User '{username}' deleted successfully.")

def create_analysis(db: Session, text: str, sentiment: str, confidence: float, model_used: str, user_id: int) -> Analysis:
    """Create a new analysis record."""
    analysis = Analysis(text=text, sentiment=sentiment, confidence=confidence, model_used=model_used, user_id=user_id)
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis

def get_analyses(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[Analysis]:
    """Retrieve a user's analysis history."""
    return db.query(Analysis).filter(Analysis.user_id == user_id).order_by(Analysis.created_at.desc()).offset(skip).limit(limit).all()

def delete_analysis(db: Session, analysis_id: int, user_id: int) -> None:
    """Delete a specific analysis record."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id, Analysis.user_id == user_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    db.delete(analysis)
    db.commit()
    logger.info(f"Analysis ID {analysis_id} deleted by user ID {user_id}.")

def update_user_preferences(db: Session, user_id: int, prefs: UserPreferenceUpdate) -> UserPreference:
    """Update user preferences."""
    db_prefs = db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
    if not db_prefs:
        db_prefs = UserPreference(user_id=user_id)
        db.add(db_prefs)
    if prefs.default_model and prefs.default_model in ALLOWED_MODELS:
        db_prefs.default_model = prefs.default_model
    if prefs.theme in ["light", "dark"]:
        db_prefs.theme = prefs.theme
    if prefs.notifications_enabled in ["yes", "no"]:
        db_prefs.notifications_enabled = prefs.notifications_enabled
    db.commit()
    db.refresh(db_prefs)
    logger.info(f"Preferences updated for user ID {user_id}.")
    return db_prefs

# --- Authentication Routes ---
@app.post("/token", response_model=Token)
async def login_for_access_token(db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()):
    """Generate access and refresh tokens for user login."""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    refresh_token = create_refresh_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer", "refresh_token": refresh_token}

@app.post("/refresh", response_model=Token)
async def refresh_access_token(refresh_token: str = Form(...), db: Session = Depends(get_db)):
    """Refresh an access token using a refresh token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid refresh token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise credentials_exception
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(db, username)
    if not user:
        raise credentials_exception
    access_token = create_access_token(data={"sub": user.username})
    new_refresh_token = create_refresh_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer", "refresh_token": new_refresh_token}

@app.post("/register", response_model=Token)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user and return tokens."""
    if get_user(db, user.username):
        raise HTTPException(status_code=400, detail="Username already registered")
    create_user(db, user)
    access_token = create_access_token(data={"sub": user.username})
    refresh_token = create_refresh_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer", "refresh_token": refresh_token}

# --- User Management Routes ---
@app.get("/users/me", response_model=Dict[str, Any])
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Retrieve current user's information."""
    return {
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "created_at": current_user.created_at,
        "last_login": current_user.last_login
    }

@app.put("/users/me", response_model=Dict[str, Any])
async def update_users_me(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update current user's information."""
    updated_user = update_user(db, current_user.username, user_update)
    return {
        "username": updated_user.username,
        "email": updated_user.email,
        "full_name": updated_user.full_name,
        "created_at": updated_user.created_at,
        "last_login": updated_user.last_login
    }

@app.delete("/users/me")
async def delete_users_me(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Delete the current user."""
    delete_user(db, current_user.username)
    return {"message": "User deleted successfully"}

@app.get("/users/preferences", response_model=Dict[str, str])
async def get_user_preferences(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Retrieve user preferences."""
    prefs = db.query(UserPreference).filter(UserPreference.user_id == current_user.id).first()
    if not prefs:
        prefs = UserPreference(user_id=current_user.id)
        db.add(prefs)
        db.commit()
        db.refresh(prefs)
    return {
        "default_model": prefs.default_model,
        "theme": prefs.theme,
        "notifications_enabled": prefs.notifications_enabled
    }

@app.put("/users/preferences", response_model=Dict[str, str])
async def update_user_preferences(
    prefs: UserPreferenceUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user preferences."""
    updated_prefs = update_user_preferences(db, current_user.id, prefs)
    return {
        "default_model": updated_prefs.default_model,
        "theme": updated_prefs.theme,
        "notifications_enabled": updated_prefs.notifications_enabled
    }

# --- Analysis Routes ---
@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze(
    text: str = Form(...),
    model: str = Form("default"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Analyze the sentiment of a single text."""
    try:
        sentiment, confidence = sentiment_service.analyze(text, model)
        analysis = create_analysis(db, text, sentiment, confidence, model, current_user.id)
        return {
            "id": analysis.id,
            "text": analysis.text,
            "sentiment": analysis.sentiment,
            "confidence": analysis.confidence,
            "model_used": analysis.model_used,
            "created_at": analysis.created_at
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/batch_analyze", response_model=List[AnalysisResponse])
async def batch_analyze(
    file: UploadFile = File(...),
    model: str = Form("default"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Analyze sentiments of multiple texts from a CSV file."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_FILE_SIZE // (1024 * 1024)}MB limit.")
    with tempfile.NamedTemporaryFile(delete=False, dir=UPLOAD_DIR, suffix=".csv") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        results = []
        with open(tmp_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            next(csv_reader, None)  # Skip header if exists
            for row in csv_reader:
                if not row or not row[0].strip():
                    continue
                text = row[0]
                sentiment, confidence = sentiment_service.analyze(text, model)
                analysis = create_analysis(db, text, sentiment, confidence, model, current_user.id)
                results.append({
                    "id": analysis.id,
                    "text": analysis.text,
                    "sentiment": analysis.sentiment,
                    "confidence": analysis.confidence,
                    "model_used": analysis.model_used,
                    "created_at": analysis.created_at
                })
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.remove(tmp_path)

@app.post("/api/batch_analyze_json", response_model=List[AnalysisResponse])
async def batch_analyze_json(
    request: BatchAnalysisRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Analyze sentiments of multiple texts from a JSON request."""
    results = []
    try:
        for text in request.texts:
            sentiment, confidence = sentiment_service.analyze(text, request.model)
            analysis = create_analysis(db, text, sentiment, confidence, request.model, current_user.id)
            results.append({
                "id": analysis.id,
                "text": analysis.text,
                "sentiment": analysis.sentiment,
                "confidence": analysis.confidence,
                "model_used": analysis.model_used,
                "created_at": analysis.created_at
            })
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/analyses", response_model=List[AnalysisResponse])
async def list_analyses(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List a user's analysis history."""
    analyses = get_analyses(db, current_user.id, skip, limit)
    return [
        {
            "id": a.id,
            "text": a.text,
            "sentiment": a.sentiment,
            "confidence": a.confidence,
            "model_used": a.model_used,
            "created_at": a.created_at
        }
        for a in analyses
    ]

@app.delete("/api/analysis/{analysis_id}")
async def delete_analysis_route(
    analysis_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a specific analysis."""
    delete_analysis(db, analysis_id, current_user.id)
    return {"message": "Analysis deleted successfully"}

# --- Model Management Routes ---
@app.get("/api/models", response_model=List[str])
async def list_models():
    """List available sentiment analysis models."""
    return list(sentiment_service.models.keys())

@app.post("/api/models/{model_name}")
async def select_model(model_name: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Select a default model for the user."""
    if model_name not in sentiment_service.models:
        raise HTTPException(status_code=400, detail="Model not available")
    prefs = update_user_preferences(db, current_user.id, UserPreferenceUpdate(default_model=model_name))
    return {"message": f"Model '{model_name}' set as default", "default_model": prefs.default_model}

# --- Health Check and Statistics ---
@app.get("/api/health")
async def health_check(db: Session = Depends(get_db)):
    """Check the health of the application."""
    try:
        db.execute("SELECT 1")
        return {"status": "healthy", "database": "connected", "models": list(sentiment_service.models.keys())}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(status_code=500, content={"status": "unhealthy", "detail": str(e)})

@app.get("/api/stats", response_model=Dict[str, Any])
async def get_user_stats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Retrieve user statistics."""
    total_analyses = db.query(Analysis).filter(Analysis.user_id == current_user.id).count()
    model_usage = db.query(Analysis.model_used, func.count(Analysis.id)).filter(Analysis.user_id == current_user.id).group_by(Analysis.model_used).all()
    return {
        "total_analyses": total_analyses,
        "model_usage": {model: count for model, count in model_usage},
        "last_login": current_user.last_login
    }

# --- UI Routes ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the home page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Render the user dashboard."""
    prefs = db.query(UserPreference).filter(UserPreference.user_id == current_user.id).first()
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "username": current_user.username, "theme": prefs.theme if prefs else "light"}
    )

# --- Error Handling ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    if request.headers.get("accept") == "application/json":
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return templates.TemplateResponse("error.html", {"request": request, "detail": exc.detail}, status_code=exc.status_code)

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    if request.headers.get("accept") == "application/json":
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
    return templates.TemplateResponse("error.html", {"request": request, "detail": "Internal server error"}, status_code=500)

# --- Application Entry Point ---
if __name__ == "__main__":
    logger.info("Starting Sentiment Analysis Tool...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
