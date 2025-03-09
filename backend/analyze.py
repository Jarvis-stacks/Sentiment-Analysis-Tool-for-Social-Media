import logging
from typing import List, Dict
from sqlalchemy.orm import Session
from database import Analysis  # Assumes database models are defined in database.py
from models import SentimentService  # Assumes SentimentService is defined in models.py
import re

# Configure logging
logger = logging.getLogger(__name__)

# Instantiate the sentiment service
sentiment_service = SentimentService()

def preprocess_text(text: str) -> str:
    """
    Preprocess the text by removing URLs, mentions, and special characters.

    Args:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove mentions (e.g., @username)
    text = re.sub(r"@\w+", "", text)
    # Remove special characters, keeping only alphanumeric and spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def analyze_single_text(text: str, model_name: str = "default") -> Dict[str, any]:
    """
    Analyze the sentiment of a single text after preprocessing.

    Args:
        text (str): The text to analyze.
        model_name (str): The name of the model to use (e.g., "default", "multilingual").

    Returns:
        Dict[str, any]: A dictionary containing the sentiment, confidence, and preprocessed text.

    Raises:
        ValueError: If the text is empty or the model name is invalid.
    """
    if not text.strip():
        logger.error("Empty text provided for analysis.")
        raise ValueError("Text cannot be empty.")
    
    preprocessed_text = preprocess_text(text)
    if not preprocessed_text:
        logger.error("Text is empty after preprocessing.")
        raise ValueError("Text is empty after preprocessing.")
    
    try:
        sentiment, confidence = sentiment_service.analyze(preprocessed_text, model_name)
        logger.info(f"Analyzed text: {text[:50]}... -> Sentiment: {sentiment}, Confidence: {confidence}")
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "preprocessed_text": preprocessed_text
        }
    except ValueError as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

def analyze_and_store(db: Session, text: str, model_name: str, user_id: int) -> Analysis:
    """
    Analyze the sentiment of a text and store the result in the database.

    Args:
        db (Session): The database session.
        text (str): The text to analyze.
        model_name (str): The name of the model to use.
        user_id (int): The ID of the user performing the analysis.

    Returns:
        Analysis: The stored analysis object.

    Raises:
        ValueError: If the text is empty or the model name is invalid.
    """
    analysis_result = analyze_single_text(text, model_name)
    analysis = Analysis(
        text=text,
        sentiment=analysis_result["sentiment"],
        confidence=analysis_result["confidence"],
        model_used=model_name,
        user_id=user_id
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    logger.info(f"Stored analysis for user {user_id}: Analysis ID {analysis.id}")
    return analysis

def analyze_batch_and_store(db: Session, texts: List[str], model_name: str, user_id: int) -> List[Analysis]:
    """
    Analyze the sentiment of a batch of texts and store the results in the database.

    Args:
        db (Session): The database session.
        texts (List[str]): The list of texts to analyze.
        model_name (str): The name of the model to use.
        user_id (int): The ID of the user performing the analysis.

    Returns:
        List[Analysis]: The list of stored analysis objects.
    """
    analyses = []
    for text in texts:
        try:
            analysis = analyze_and_store(db, text, model_name, user_id)
            analyses.append(analysis)
        except ValueError as e:
            logger.warning(f"Skipping text due to error: {str(e)}")
    return analyses

def get_user_analyses(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[Analysis]:
    """
    Retrieve a user's analysis history from the database.

    Args:
        db (Session): The database session.
        user_id (int): The ID of the user.
        skip (int): Number of records to skip (for pagination).
        limit (int): Maximum number of records to return.

    Returns:
        List[Analysis]: The list of analysis objects for the user.
    """
    return db.query(Analysis).filter(Analysis.user_id == user_id).offset(skip).limit(limit).all()
