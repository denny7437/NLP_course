#!/usr/bin/env python3
"""
Безопасный загрузчик обученной модели
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import os

class SafeReligiousDetector:
    def __init__(self, model_path="./religious_detector_manual"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cpu')
        
    def load_model(self):
        """Безопасная загрузка модели"""
        print(f"📂 Загружаю модель из {self.model_path}...")
        
        # Загрузка токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Загрузка конфигурации
        config = AutoConfig.from_pretrained(self.model_path)
        
        # Создание модели с правильной конфигурацией
        self.model = AutoModelForSequenceClassification.from_config(config)
        
        # Загрузка весов
        state_dict_path = os.path.join(self.model_path, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            print("✅ Веса модели загружены успешно!")
        else:
            print("⚠️ Файл весов не найден, используется неинициализированная модель")
            
        self.model.to(self.device)
        self.model.eval()
        print("🎯 Модель готова к использованию!")
    
    def predict(self, text):
        """Предсказание для текста"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Модель не загружена. Вызовите load_model()")
            
        # Токенизация
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to(self.device)
        
        # Предсказание
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1).item()
        
        return prediction, probs[0].cpu().numpy()
    
    def batch_predict(self, texts):
        """Предсказание для списка текстов"""
        predictions = []
        probabilities = []
        
        for text in texts:
            pred, prob = self.predict(text)
            predictions.append(pred)
            probabilities.append(prob)
            
        return predictions, probabilities

def test_trained_model():
    """Тестирование обученной модели"""
    print("🧪 ТЕСТИРОВАНИЕ ОБУЧЕННОЙ МОДЕЛИ")
    print("="*50)
    
    # Создание детектора
    detector = SafeReligiousDetector()
    detector.load_model()
    
    # Тестовые примеры
    test_cases = [
        ("Религия помогает людям найти смысл жизни", 0),  # Безопасный
        ("Все верующие - дураки и фанатики", 1),  # Опасный
        ("Духовные практики могут быть полезными", 0),  # Безопасный
        ("Церкви должны быть сожжены дотла", 1),  # Опасный
        ("Каждый имеет право на свою веру", 0),  # Безопасный
        ("Религия - это яд для общества", 1),  # Опасный
        ("Молитва приносит внутренний покой", 0),  # Безопасный
        ("Религиозные лидеры должны быть уничтожены", 1),  # Опасный
    ]
    
    correct = 0
    total = len(test_cases)
    
    print("\n🔍 Результаты тестирования:")
    
    for i, (text, expected) in enumerate(test_cases, 1):
        pred, prob = detector.predict(text)
        
        safe_prob = prob[0] * 100
        dangerous_prob = prob[1] * 100
        confidence = max(prob) * 100
        
        print(f"\n{i}. {text}")
        print(f"   Ожидалось: {'Безопасный' if expected == 0 else 'Опасный'}")
        print(f"   Предсказано: {'Безопасный' if pred == 0 else 'Опасный'}")
        print(f"   Вероятности: Безопасный={safe_prob:.1f}%, Опасный={dangerous_prob:.1f}%")
        print(f"   Уверенность: {confidence:.1f}%")
        
        if pred == expected:
            print("   ✅ Правильно!")
            correct += 1
        else:
            print("   ❌ Неправильно!")
    
    accuracy = correct / total
    print(f"\n📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   Правильных ответов: {correct}/{total}")
    print(f"   Точность: {accuracy:.1%}")
    
    if accuracy >= 0.9:
        print("   🏆 Превосходная работа модели!")
    elif accuracy >= 0.8:
        print("   🎉 Отличная работа модели!")
    elif accuracy >= 0.7:
        print("   👍 Хорошая работа модели!")
    else:
        print("   ⚠️ Модель требует улучшения")

if __name__ == "__main__":
    test_trained_model() 