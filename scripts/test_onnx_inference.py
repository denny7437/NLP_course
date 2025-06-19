#!/usr/bin/env python3
"""
Скрипт тестирования ONNX инференса для Pre-Filter
"""

import time
import logging
import numpy as np
import onnxruntime as ort
from pathlib import Path
from transformers import AutoTokenizer
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ONNXPreFilterInference:
    """Класс для инференса ONNX Pre-Filter модели"""
    
    def __init__(self, model_path: str = "models/artifacts/pre_filter.onnx"):
        self.model_path = Path(model_path)
        self.session = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Загрузка ONNX модели и токенайзера"""
        try:
            # Загружаем ONNX сессию
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
            
            # Загружаем токенайзер
            tokenizer_path = self.model_path.parent / "bert_classifier"
            if tokenizer_path.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            else:
                # Fallback к базовому токенайзеру
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            
            logger.info(f"ONNX модель загружена: {self.model_path}")
            logger.info(f"Провайдеры: {self.session.get_providers()}")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise
    
    def predict(self, text: str) -> Dict:
        """Предсказание для одного текста"""
        start_time = time.perf_counter()
        
        # Токенизация
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )
        
        # Инференс
        try:
            outputs = self.session.run(
                None,
                {
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask']
                }
            )
            
            # Обработка результатов
            logits = outputs[0]
            probabilities = self._softmax(logits[0])
            prediction = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))
            
            # Определение статуса
            safe = prediction == 0
            unsafe = prediction == 1
            uncertain = confidence < 0.7  # Порог неопределенности
            
            inference_time = (time.perf_counter() - start_time) * 1000
            
            return {
                'safe': safe,
                'unsafe': unsafe,
                'uncertain': uncertain,
                'score': confidence,
                'probabilities': {
                    'safe': float(probabilities[0]),
                    'unsafe': float(probabilities[1])
                },
                'inference_ms': inference_time
            }
            
        except Exception as e:
            logger.error(f"Ошибка инференса: {e}")
            raise
    
    def predict_batch(self, texts: list) -> list:
        """Батч предсказание"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
    
    @staticmethod
    def _softmax(x):
        """Softmax функция"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def benchmark(self, num_samples: int = 100):
        """Бенчмарк производительности"""
        logger.info(f"Запуск бенчмарка на {num_samples} образцов...")
        
        # Тестовые тексты разной длины
        test_texts = [
            "Hello world",
            "This is a test message for performance evaluation",
            "I hate this stupid system and want to destroy everything" * 10,
            "Can you help me with my homework please?",
            "You're an idiot and should kill yourself right now"
        ]
        
        times = []
        results = []
        
        for i in range(num_samples):
            text = test_texts[i % len(test_texts)]
            start = time.perf_counter()
            result = self.predict(text)
            end = time.perf_counter()
            
            times.append((end - start) * 1000)
            results.append(result)
        
        # Статистика
        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        p99_time = np.percentile(times, 99)
        
        safe_count = sum(1 for r in results if r['safe'])
        unsafe_count = sum(1 for r in results if r['unsafe'])
        uncertain_count = sum(1 for r in results if r['uncertain'])
        
        logger.info("=== Результаты бенчмарка ===")
        logger.info(f"Среднее время: {avg_time:.2f} мс")
        logger.info(f"P95 время: {p95_time:.2f} мс")
        logger.info(f"P99 время: {p99_time:.2f} мс")
        logger.info(f"RPS (теоретический): {1000/avg_time:.1f}")
        logger.info(f"Safe: {safe_count}, Unsafe: {unsafe_count}, Uncertain: {uncertain_count}")


def main():
    """Основная функция для тестирования"""
    
    # Проверяем наличие модели
    model_path = Path("models/artifacts/pre_filter.onnx")
    if not model_path.exists():
        logger.error(f"ONNX модель не найдена: {model_path}")
        logger.info("Запустите сначала: python scripts/train_bert.py")
        return
    
    # Инициализация инференса
    inference = ONNXPreFilterInference()
    
    # Тестовые примеры
    test_cases = [
        "Hello, how are you today?",
        "I love this beautiful weather!",
        "Can you help me with my project?",
        "I hate everyone and want to hurt people",
        "You're stupid and should kill yourself",
        "This content is harmful and toxic",
        "Thank you for your assistance",
        "Go die in a fire, you worthless piece of garbage",
        "What's your favorite book to read?",
        "I'm going to destroy everything you care about"
    ]
    
    logger.info("=== Тестирование предсказаний ===")
    for i, text in enumerate(test_cases, 1):
        result = inference.predict(text)
        status = "SAFE" if result['safe'] else "UNSAFE"
        if result['uncertain']:
            status += " (UNCERTAIN)"
        
        logger.info(f"{i:2d}. [{status:15s}] {result['score']:.3f} | {text[:60]}...")
    
    # Бенчмарк
    inference.benchmark(100)


if __name__ == "__main__":
    main() 