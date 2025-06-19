"""
Pre-Filter Service - BERT ONNX модель для быстрой классификации
"""

import time
import logging
import asyncio
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Optional
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class PreFilterService:
    """Сервис быстрой фильтрации с BERT ONNX моделью"""
    
    def __init__(self, 
                 model_path: str = "models/artifacts/pre_filter.onnx",
                 uncertainty_threshold: float = 0.7,
                 max_workers: int = 4):
        self.model_path = Path(model_path)
        self.uncertainty_threshold = uncertainty_threshold
        self.session = None
        self.tokenizer = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._initialize()
    
    def _initialize(self):
        """Инициализация модели и токенайзера"""
        try:
            # Создаем ONNX сессию с оптимизацией для CPU
            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = 2
            sess_options.intra_op_num_threads = 2
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(
                str(self.model_path), 
                sess_options=sess_options,
                providers=providers
            )
            
            # Загружаем токенайзер
            tokenizer_path = self.model_path.parent / "bert_classifier"
            if tokenizer_path.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            else:
                logger.warning("Токенайзер не найден, используется базовый DistilBERT")
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            
            logger.info(f"Pre-Filter инициализирован: {self.model_path}")
            logger.info(f"Провайдеры ONNX: {self.session.get_providers()}")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации Pre-Filter: {e}")
            raise
    
    def _tokenize(self, text: str) -> Dict[str, np.ndarray]:
        """Токенизация текста"""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
    
    def _run_inference(self, inputs: Dict[str, np.ndarray]) -> Dict:
        """Запуск инференса модели"""
        try:
            # ONNX инференс
            outputs = self.session.run(None, inputs)
            logits = outputs[0][0]  # Извлекаем первый элемент батча
            
            # Softmax для получения вероятностей
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)
            
            # Предсказание и уверенность
            prediction = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': {
                    'safe': float(probabilities[0]),
                    'unsafe': float(probabilities[1])
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка инференса: {e}")
            raise
    
    def predict_sync(self, text: str) -> Dict:
        """Синхронное предсказание"""
        start_time = time.perf_counter()
        
        # Валидация входа
        if not text or len(text.strip()) == 0:
            return {
                'safe': True,
                'unsafe': False,
                'uncertain': False,
                'score': 1.0,
                'inference_ms': 0.0
            }
        
        # Токенизация
        inputs = self._tokenize(text)
        
        # Инференс
        result = self._run_inference(inputs)
        
        # Определение статуса
        prediction = result['prediction']
        confidence = result['confidence']
        
        safe = prediction == 0
        unsafe = prediction == 1
        uncertain = confidence < self.uncertainty_threshold
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        return {
            'safe': safe,
            'unsafe': unsafe,
            'uncertain': uncertain,
            'score': confidence,
            'probabilities': result['probabilities'],
            'inference_ms': inference_time
        }
    
    async def predict(self, text: str) -> Dict:
        """Асинхронное предсказание"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor, 
            self.predict_sync, 
            text
        )
        return result
    
    async def predict_batch(self, texts: list) -> list:
        """Батч предсказание"""
        tasks = [self.predict(text) for text in texts]
        results = await asyncio.gather(*tasks)
        return results
    
    def health_check(self) -> Dict:
        """Проверка здоровья сервиса"""
        try:
            # Тестовый инференс
            test_result = self.predict_sync("Test message")
            
            return {
                'status': 'healthy',
                'model_loaded': self.session is not None,
                'tokenizer_loaded': self.tokenizer is not None,
                'test_inference_ms': test_result['inference_ms']
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'model_loaded': self.session is not None,
                'tokenizer_loaded': self.tokenizer is not None
            }
    
    def get_stats(self) -> Dict:
        """Получение статистики сервиса"""
        return {
            'model_path': str(self.model_path),
            'uncertainty_threshold': self.uncertainty_threshold,
            'max_workers': self.executor._max_workers,
            'providers': self.session.get_providers() if self.session else None
        }
    
    def close(self):
        """Закрытие ресурсов"""
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("Pre-Filter сервис закрыт") 