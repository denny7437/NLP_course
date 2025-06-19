#!/usr/bin/env python3
"""
Тестирование детектора оскорбления чувств верующих
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os

class ReligiousContentTester:
    def __init__(self, model_path="./religious_detector_manual"):
        """
        Инициализация тестера
        
        Args:
            model_path: путь к обученной модели (по умолчанию финальная модель)
        """
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cpu')  # Используем CPU для стабильности
    
    def initialize_model(self):
        """Загрузка модели и токенизатора"""
        print(f"🔄 Загружаю модель из {self.model_path}...")
        
        # Проверяем, существует ли обученная модель
        if os.path.exists(self.model_path):
            print("✅ Найдена обученная модель!")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                print("🎯 Загружена ОБУЧЕННАЯ модель для детекции религиозного контента")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки обученной модели: {e}")
                print("🔄 Переключаюсь на базовую модель...")
                self._load_base_model()
        else:
            print("⚠️ Обученная модель не найдена, загружаю базовую модель...")
            self._load_base_model()
        
        self.model.to(self.device)
        self.model.eval()
        print("✅ Модель готова к работе!")
    
    def _load_base_model(self):
        """Загрузка базовой модели ModernBERT"""
        model_name = "answerdotai/ModernBERT-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,
            problem_type="single_label_classification"
        )
        print("🤖 Загружена базовая ModernBERT (необученная)")
    
    def predict(self, texts):
        """
        Предсказание для списка текстов
        
        Args:
            texts: список текстов для анализа
            
        Returns:
            tuple: (predictions, probabilities)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Модель не инициализирована. Вызовите initialize_model()")
        
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for text in texts:
                # Токенизация
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                ).to(self.device)
                
                # Предсказание
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Применяем softmax для получения вероятностей
                probs = torch.softmax(logits, dim=-1)
                probs_np = probs.cpu().numpy()[0]
                
                # Получаем предсказанный класс
                predicted_class = torch.argmax(logits, dim=-1).item()
                
                predictions.append(predicted_class)
                probabilities.append(probs_np)
        
        return predictions, probabilities
    
    def test_examples(self):
        """Тестирование на примерах"""
        print("\n🧪 ТЕСТИРОВАНИЕ НА ПРИМЕРАХ")
        print("="*50)
        
        # Примеры для тестирования
        test_cases = [
            ("Религия помогает людям найти смысл жизни", 0),  # Безопасный
            ("Все верующие - дураки и фанатики", 1),  # Опасный
            ("Духовные практики могут быть полезными", 0),  # Безопасный
            ("Церкви должны быть сожжены дотла", 1),  # Опасный
            ("Каждый имеет право на свою веру", 0),  # Безопасный
            ("Религия - это яд для общества", 1),  # Опасный
        ]
        
        correct = 0
        total = len(test_cases)
        
        for i, (text, expected) in enumerate(test_cases, 1):
            predictions, probabilities = self.predict([text])
            predicted = predictions[0]
            prob = probabilities[0]
            
            print(f"\n{i}. Текст: {text}")
            print(f"   Ожидаемый класс: {expected} ({'Безопасный' if expected == 0 else 'Опасный'})")
            print(f"   Предсказанный: {predicted} ({'Безопасный' if predicted == 0 else 'Опасный'})")
            print(f"   Вероятности: Безопасный={prob[0]:.3f}, Опасный={prob[1]:.3f}")
            
            if predicted == expected:
                print("   ✅ Правильно!")
                correct += 1
            else:
                print("   ❌ Неправильно!")
        
        accuracy = correct / total
        print(f"\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
        print(f"   Правильных ответов: {correct}/{total}")
        print(f"   Точность: {accuracy:.1%}")
        
        if accuracy >= 0.8:
            print("   🎉 Отличная работа модели!")
        elif accuracy >= 0.6:
            print("   👍 Неплохая работа модели")
        else:
            print("   ⚠️ Модель требует дообучения")

def main():
    """Основная функция для тестирования"""
    tester = ReligiousContentTester()
    tester.initialize_model()
    tester.test_examples()

if __name__ == "__main__":
    main()