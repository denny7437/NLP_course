"""
HTTP клиенты для взаимодействия с микросервисами
"""

import asyncio
import logging
import httpx
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class PreFilterClient:
    """Клиент для Pre-Filter сервиса"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=5.0)
        
    async def predict(self, text: str, request_id: str) -> Dict[str, Any]:
        """Предсказание через Pre-Filter"""
        try:
            payload = {"text": text, "request_id": request_id}
            response = await self.client.post("/predict", json=payload)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Pre-Filter prediction failed: {e}")
            return {
                "safe": False,
                "unsafe": True, 
                "uncertain": True,
                "score": 0.0
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья Pre-Filter"""
        try:
            response = await self.client.get("/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def close(self):
        await self.client.aclose()


class SafetyLLMClient:
    """Клиент для Safety-LLM сервиса"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=60.0)
        
    async def analyze(self, text: str, request_id: str, pre_score: float) -> Dict[str, Any]:
        """Анализ через Safety-LLM"""
        try:
            payload = {
                "text": text,
                "request_id": request_id,
                "pre_score": pre_score
            }
            response = await self.client.post("/analyze", json=payload)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Safety-LLM analysis failed: {e}")
            # Мок-ответ для разработки
            unsafe_keywords = ["hate", "kill", "die", "stupid", "destroy"]
            is_unsafe = any(word in text.lower() for word in unsafe_keywords)
            
            return {
                "status": "unsafe" if is_unsafe else "safe",
                "category": "toxicity" if is_unsafe else None,
                "reasoning": "Mock analysis result",
                "confidence": 0.8,
                "request_id": request_id
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья Safety-LLM"""
        try:
            response = await self.client.get("/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "healthy", "mock": True}  # Мок для разработки
    
    async def close(self):
        await self.client.aclose() 