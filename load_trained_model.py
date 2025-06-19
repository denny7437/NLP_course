#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ReligiousDetector:
    def __init__(self, model_path="./religious_detector_manual"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
    
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
        
        return prediction, probs[0].numpy()

# Пример использования
if __name__ == "__main__":
    detector = ReligiousDetector()
    
    examples = [
        "Религия помогает людям найти смысл жизни",
        "Все верующие - дураки и фанатики",
        "Духовные практики могут быть полезными",
        "Церкви должны быть сожжены дотла"
    ]
    
    print("🧪 Тестирование обученной модели:")
    for text in examples:
        pred, prob = detector.predict(text)
        class_name = "Безопасный" if pred == 0 else "Опасный"
        confidence = prob[pred]
        
        print(f"\nТекст: {text}")
        print(f"Класс: {class_name}")
        print(f"Уверенность: {confidence:.3f}")
