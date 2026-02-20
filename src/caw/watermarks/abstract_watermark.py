from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..client import get_openai_client
from ..config import get_config


class AbstractWatermark(ABC):
    """Abstract base class for watermarking code agent trajectories."""

    def __init__(
        self,
        model: Optional[str] = None,
        openai_api_key: str = "EMPTY",
        openai_api_base: str = "http://localhost:8000/v1",
    ):
        self.client = get_openai_client(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.config = get_config()
        self.model = model or self.config.watermark_model
    
    @abstractmethod
    def check(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Check if watermark can be applied to the given messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            True if watermark can be applied, False otherwise
        """
        pass
    
    @abstractmethod
    def inject(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Inject watermark into the messages.
        
        Args:
            messages: List of message dictionaries to watermark
            
        Returns:
            Watermarked messages
        """
        pass
    
    def detect(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Detect if messages contain this watermark.
        
        Args:
            messages: List of message dictionaries to check
            
        Returns:
            True if watermark is detected
        """
        return False
    
    def validate_messages(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Validate message format.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            True if all messages have required format
        """
        if not isinstance(messages, list):
            return False
            
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if 'role' not in msg or 'content' not in msg:
                return False
            if msg['role'] not in ['system', 'user', 'assistant', 'tool']:
                return False
                
        return True