# Import necessary modules
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, Form, Request, status, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from transformers import pipeline
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
import csv
from io import StringIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("sentiment_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Application configuration
DATABASE_URL = "sqlite:///./sentiment.db"
SECRET_KEY = os.getenv("SECRET_KEY", "your-very-secure-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
ALLOWED_MODELS = ["default", "distilbert"]

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis Tool",
    description="A comprehensive tool for analyzing text sentiment using Hugging Face models.",
    version="1.0.0"
)

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Database setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class User(Base):
    """
    Database model for users.
    
    Attributes:
        id (int): Primary key for the user.
        username (str): Unique username.
        hashed_password (str): Hashed password for security.
        analyses (relationship): Relationship to user's analyses.
    """
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    analyses = relationship("Analysis", back_populates="user")

class Analysis(Base):
    """
    Database model for sentiment analyses.
    
    Attributes:
        id (int): Primary key for the analysis.
        text (str): Input text analyzed.
        sentiment (str): Predicted sentiment (e.g., POSITIVE, NEGATIVE).
        confidence (float): Confidence score of the prediction.
        model_used (str): Model used for analysis.
        user_id (int): Foreign key to the user who performed the analysis.
        user (relationship): Relationship to the user.
    """
    __tablename__ = "analyses"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    sentiment = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    model_used = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="analyses")

# Create database tables
Base.metadata.create_all(bind=engine)

# Pydantic Models for API validation
class Token(BaseModel):
    """Model for authentication token response."""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Model for token payload data."""
    username: Optional[str] = None

class UserCreate(BaseModel):
    """Model for user registration input."""
    username: str
    password: str

class UserInDB(BaseModel):
    """Model for user data in the database."""
    username: str
    hashed_password: str

class AnalysisResponse(BaseModel):
    """Model for single analysis response."""
    text: str
    sentiment: str
    confidence: float
    model_used: str

# Dependency to get database session
def get_db():
    """
    Dependency function to provide a database session.
    
    Yields:
        Session: SQLAlchemy session object.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        logger.debug("Database session closed.")

# Authentication setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password.
    
    Args:
        plain_password (str): The password to verify.
        hashed_password (str): The stored hashed password.
    
    Returns:
        bool: True if the password matches, False otherwise.
    """
    logger.debug(f"Verifying password for hashed_password: {hashed_password[:10]}...")
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """
    Hash a plain password.
    
    Args:
        password (str): The password to hash.
    
    Returns:
        str: The hashed password.
    """
    logger.debug("Generating password hash.")
    return pwd_context.hash(password)

def get_user(db: Session, username: str) -> Optional[User]:
    """
    Retrieve a user from the database by username.
    
    Args:
        db (Session): Database session.
        username (str): Username to search for.
    
    Returns:
        Optional[User]: User object if found, None otherwise.
    """
    logger.debug(f"Fetching user: {username}")
    return db.query(User).filter(User.username == username).first()

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """
    Authenticate a user with username and password.
    
    Args:
        db (Session): Database session.
        username (str): User's username.
        password (str): User's password.
    
    Returns:
        Optional[User]: Authenticated user object, or None if authentication fails.
    """
    logger.info(f"Attempting to authenticate user: {username}")
    user = get_user(db, username)
    if not user or not verify_password(password, user.hashed_password):
        logger.warning(f"Authentication failed for user: {username}")
        return None
    logger.info(f"User {username} authenticated successfully.")
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data (dict): Data to encode in the token.
        expires_delta (Optional[timedelta]): Token expiration time delta.
    
    Returns:
        str: Encoded JWT token.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    logger.debug(f"Created access token expiring at: {expire}")
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """
    Get the current authenticated user from the token.
    
    Args:
        token (str): JWT token from the request.
        db (Session): Database session.
    
    Returns:
        User: Authenticated user object.
    
    Raises:
        HTTPException: If token is invalid or user not found.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            logger.error("Token payload missing 'sub' field.")
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError as e:
        logger.error(f"JWT decode error: {str(e)}")
        raise credentials_exception
    user = get_user(db, username=token_data.username)
    if user is None:
        logger.error(f"User not found for username: {token_data.username}")
        raise credentials_exception
    logger.debug(f"Current user retrieved: {user.username}")
    return user

# Sentiment Analysis Service
class SentimentService:
    """
    Service class for handling sentiment analysis using Hugging Face models.
    
    Attributes:
        models (dict): Dictionary of loaded sentiment analysis models.
    """
    def __init__(self):
        """Initialize the sentiment service with pre-trained models."""
        logger.info("Initializing SentimentService with pre-trained models.")
        self.models = {
            "default": pipeline("sentiment-analysis"),
            "distilbert": pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"),
        }
        logger.info("Sentiment models loaded successfully.")

    def analyze(self, text: str, model_name: str = "default") -> tuple[str, float]:
        """
        Analyze the sentiment of a given text.
        
        Args:
            text (str): Text to analyze.
            model_name (str): Name of the model to use.
        
        Returns:
            tuple[str, float]: Sentiment label and confidence score.
        
        Raises:
            ValueError: If the model name is invalid.
        """
        if model_name not in self.models:
            logger.error(f"Invalid model name requested: {model_name}")
            raise ValueError(f"Model '{model_name}' not available. Choose from {list(self.models.keys())}")
        logger.info(f"Analyzing text with model: {model_name}")
        result = self.models[model_name](text)[0]
        sentiment = result["label"]
        confidence = result["score"]
        logger.debug(f"Analysis result - Text: {text[:50]}..., Sentiment: {sentiment}, Confidence: {confidence}")
        return sentiment, confidence

# Instantiate the sentiment service
sentiment_service = SentimentService()

# Authentication Routes
@app.post("/token", response_model=Token)
async def login_for_access_token(db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Endpoint to generate an access token for authenticated users.
    
    Args:
        db (Session): Database session.
        form_data (OAuth2PasswordRequestForm): Login form data.
    
    Returns:
        dict: Token response with access_token and token_type.
    
    Raises:
        HTTPException: If authentication fails.
    """
    logger.info(f"Login attempt for user: {form_data.username}")
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    logger.info(f"Token generated for user: {user.username}")
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register", response_class=HTMLResponse)
async def register(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Endpoint to register a new user.
    
    Args:
        request (Request): FastAPI request object.
        username (str): New user's username.
        password (str): New user's password.
        db (Session): Database session.
    
    Returns:
        HTMLResponse: Registration success or failure page.
    
    Raises:
        HTTPException: If username is already taken.
    """
    logger.info(f"Registration attempt for username: {username}")
    existing_user = get_user(db, username)
    if existing_user:
        logger.warning(f"Username already registered: {username}")
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Username already registered"}
        )
    hashed_password = get_password_hash(password)
    new_user = User(username=username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    logger.info(f"User registered successfully: {username}")
    return templates.TemplateResponse(
        "register.html",
        {"request": request, "message": "Registration successful! Please log in."}
    )

# Frontend Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Home page of the sentiment analysis tool.
    
    Args:
        request (Request): FastAPI request object.
    
    Returns:
        HTMLResponse: Rendered home page template.
    """
    logger.debug("Serving home page.")
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """
    Login page for user authentication.
    
    Args:
        request (Request): FastAPI request object.
    
    Returns:
        HTMLResponse: Rendered login page template.
    """
    logger.debug("Serving login page.")
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """
    Registration page for new users.
    
    Args:
        request (Request): FastAPI request object.
    
    Returns:
        HTMLResponse: Rendered registration page template.
    """
    logger.debug("Serving register page.")
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/analyze", response_class=HTMLResponse)
async def analyze_page(request: Request, current_user: User = Depends(get_current_user)):
    """
    Page for single text sentiment analysis.
    
    Args:
        request (Request): FastAPI request object.
        current_user (User): Authenticated user.
    
    Returns:
        HTMLResponse: Rendered analysis page template.
    """
    logger.debug(f"Serving analyze page for user: {current_user.username}")
    return templates.TemplateResponse(
        "analyze.html",
        {"request": request, "models": ALLOWED_MODELS}
    )

@app.get("/batch", response_class=HTMLResponse)
async def batch_page(request: Request, current_user: User = Depends(get_current_user)):
    """
    Page for batch sentiment analysis.
    
    Args:
        request (Request): FastAPI request object.
        current_user (User): Authenticated user.
    
    Returns:
        HTMLResponse: Rendered batch analysis page template.
    """
    logger.debug(f"Serving batch analysis page for user: {current_user.username}")
    return templates.TemplateResponse("batch.html", {"request": request})

@app.get("/history", response_class=HTMLResponse)
async def history_page(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Page to view user's analysis history.
    
    Args:
        request (Request): FastAPI request object.
        current_user (User): Authenticated user.
        db (Session): Database session.
    
    Returns:
        HTMLResponse: Rendered history page template with analysis data.
    """
    logger.debug(f"Serving history page for user: {current_user.username}")
    analyses = db.query(Analysis).filter(Analysis.user_id == current_user.id).all()
    return templates.TemplateResponse(
        "history.html",
        {"request": request, "analyses": analyses}
    )

@app.get("/profile", response_class=HTMLResponse)
async def profile_page(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """
    User profile page.
    
    Args:
        request (Request): FastAPI request object.
        current_user (User): Authenticated user.
    
    Returns:
        HTMLResponse: Rendered profile page template.
    """
    logger.debug(f"Serving profile page for user: {current_user.username}")
    return templates.TemplateResponse(
        "profile.html",
        {"request": request, "user": current_user}
    )

# API Routes
@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze(
    text: str = Form(...),
    model: str = Form("default"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    API endpoint to analyze sentiment of a single text.
    
    Args:
        text (str): Text to analyze.
        model (str): Model to use for analysis.
        db (Session): Database session.
        current_user (User): Authenticated user.
    
    Returns:
        AnalysisResponse: Sentiment analysis result.
    
    Raises:
        HTTPException: If model is invalid or analysis fails.
    """
    logger.info(f"Analyzing text for user: {current_user.username} with model: {model}")
    try:
        sentiment, confidence = sentiment_service.analyze(text, model)
        analysis = Analysis(
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            model_used=model,
            user_id=current_user.id
        )
        db.add(analysis)
        db.commit()
        logger.info(f"Analysis saved for user: {current_user.username}")
        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "model_used": model
        }
    except ValueError as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/batch_analyze", response_model=List[AnalysisResponse])
async def batch_analyze(
    file: UploadFile = File(...),
    model: str = Form("default"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    API endpoint to analyze sentiment of multiple texts from a file.
    
    Args:
        file (UploadFile): Uploaded file containing texts (CSV format).
        model (str): Model to use for analysis.
        db (Session): Database session.
        current_user (User): Authenticated user.
    
    Returns:
        List[AnalysisResponse]: List of sentiment analysis results.
    
    Raises:
        HTTPException: If file format is invalid or analysis fails.
    """
    logger.info(f"Batch analysis requested by user: {current_user.username} with model: {model}")
    if not file.filename.endswith('.csv'):
        logger.error("Invalid file format. Only CSV files are supported.")
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    contents = await file.read()
    text_data = contents.decode("utf-8")
    csv_reader = csv.reader(StringIO(text_data))
    results = []
    
    try:
        for row in csv_reader:
            if not row or not row[0].strip():
                continue  # Skip empty rows
            text = row[0]
            sentiment, confidence = sentiment_service.analyze(text, model)
            analysis = Analysis(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                model_used=model,
                user_id=current_user.id
            )
            db.add(analysis)
            results.append({
                "text": text,
                "sentiment": sentiment,
                "confidence": confidence,
                "model_used": model
            })
        db.commit()
        logger.info(f"Batch analysis completed for {len(results)} texts.")
        return results
    except ValueError as e:
        logger.error(f"Batch analysis error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during batch analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/analyses", response_model=List[AnalysisResponse])
async def list_analyses(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    API endpoint to list all analyses for the current user.
    
    Args:
        db (Session): Database session.
        current_user (User): Authenticated user.
    
    Returns:
        List[AnalysisResponse]: List of user's analyses.
    """
    logger.debug(f"Fetching analysis history for user: {current_user.username}")
    analyses = db.query(Analysis).filter(Analysis.user_id == current_user.id).all()
    return [
        {
            "text": analysis.text,
            "sentiment": analysis.sentiment,
            "confidence": analysis.confidence,
            "model_used": analysis.model_used
        }
        for analysis in analyses
    ]

@app.delete("/api/analysis/{analysis_id}")
async def delete_analysis(
    analysis_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    API endpoint to delete a specific analysis.
    
    Args:
        analysis_id (int): ID of the analysis to delete.
        db (Session): Database session.
        current_user (User): Authenticated user.
    
    Returns:
        dict: Success message.
    
    Raises:
        HTTPException: If analysis is not found or not owned by the user.
    """
    logger.info(f"Delete request for analysis ID: {analysis_id} by user: {current_user.username}")
    analysis = db.query(Analysis).filter(
        Analysis.id == analysis_id,
        Analysis.user_id == current_user.id
    ).first()
    if not analysis:
        logger.error(f"Analysis ID: {analysis_id} not found or not owned by user.")
        raise HTTPException(status_code=404, detail="Analysis not found")
    db.delete(analysis)
    db.commit()
    logger.info(f"Analysis ID: {analysis_id} deleted successfully.")
    return {"message": "Analysis deleted successfully"}

# Run the application (for local testing)
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
