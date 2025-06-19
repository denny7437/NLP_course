#!/usr/bin/env python3
"""
Ручное сохранение обученной модели
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import os

class SimpleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_and_save_manually():
    print("🔥 Ручное обучение и сохранение ModernBERT")
    
    # Загрузка данных
    df = pd.read_csv("train_diff - Лист1.csv")
    print(f"📊 Загружено {len(df)} примеров")
    
    # Анализ распределения
    safe_count = (df['label'] == 0).sum()
    dangerous_count = (df['label'] == 1).sum()
    print(f"✅ Безопасных: {safe_count} ({safe_count/len(df)*100:.1f}%)")
    print(f"🚨 Опасных: {dangerous_count} ({dangerous_count/len(df)*100:.1f}%)")
    
    # Разделение данных
    X_train, X_val, y_train, y_val = train_test_split(
        df['text'].values, df['label'].values, 
        test_size=0.2, random_state=42, stratify=df['label']
    )
    
    print(f"🔄 Тренировочных: {len(X_train)}, Валидационных: {len(X_val)}")
    
    # Инициализация модели
    model_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    
    device = torch.device('cpu')
    model.to(device)
    
    # Создание датасетов
    train_dataset = SimpleDataset(X_train, y_train, tokenizer)
    val_dataset = SimpleDataset(X_val, y_val, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Оптимизатор
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Обучение
    model.train()
    num_epochs = 2
    
    print("🚀 Начинаю обучение...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Вычисляем точность
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 25 == 0:
                print(f"Эпоха {epoch+1}/{num_epochs}, Батч {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        print(f"✅ Эпоха {epoch+1}: Loss={avg_loss:.4f}, Точность={accuracy:.3f}")
        
        # Валидация
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        val_accuracy = val_correct / val_total
        print(f"📊 Валидационная точность: {val_accuracy:.3f}")
        model.train()
    
    # Ручное сохранение модели
    save_dir = "./religious_detector_manual"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"💾 Сохраняю модель в {save_dir}...")
    
    # Сохранение state_dict модели
    torch.save(model.state_dict(), f"{save_dir}/pytorch_model.bin")
    
    # Сохранение конфигурации
    model.config.save_pretrained(save_dir)
    
    # Сохранение токенизатора
    tokenizer.save_pretrained(save_dir)
    
    print("✅ Модель сохранена!")
    
    # Создание простого загрузчика
    loader_code = '''#!/usr/bin/env python3
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
        
        print(f"\\nТекст: {text}")
        print(f"Класс: {class_name}")
        print(f"Уверенность: {confidence:.3f}")
'''
    
    with open("load_trained_model.py", "w") as f:
        f.write(loader_code)
    
    print("📝 Создан load_trained_model.py для использования")
    
    # Быстрый тест
    print("\n🧪 Быстрый тест...")
    model.eval()
    test_texts = ["Религия помогает людям", "Все верующие дураки"]
    
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs.to(device))
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(outputs.logits, dim=-1).item()
        
        print(f"'{text}' -> {pred} ({'Безопасный' if pred == 0 else 'Опасный'}) [{probs[0][pred]:.3f}]")
    
    print("🎉 Готово! Модель обучена и сохранена!")

if __name__ == "__main__":
    train_and_save_manually() 