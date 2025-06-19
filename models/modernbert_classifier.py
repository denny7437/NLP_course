"""
ModernBERT Classifier для обучения и классификации текста
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset as HFDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import json
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Конфигурация для обучения ModernBERT"""
    model_name: str = "answerdotai/ModernBERT-base"
    num_labels: int = 2
    max_length: int = 8192  # ModernBERT поддерживает до 8192 токенов
    batch_size: int = 8     # Уменьшили для длинного контекста
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    output_dir: str = "./models/modernbert_classifier"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    early_stopping_patience: int = 3


class TextClassificationDataset(Dataset):
    """Dataset для классификации текста"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
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
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ModernBERTClassifier:
    """Класс для обучения и использования ModernBERT классификатора"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def initialize_model(self):
        """Инициализация модели и токенизатора"""
        logger.info(f"Загружаю модель {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            reference_compile=False  # Отключаем компиляцию для избежания проблем с CUDA
        )
        
        # Добавляем pad_token если его нет
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("Модель и токенизатор загружены успешно")
    
    def prepare_datasets(self, 
                        train_texts: List[str], 
                        train_labels: List[int],
                        val_texts: Optional[List[str]] = None,
                        val_labels: Optional[List[int]] = None) -> Tuple[HFDataset, Optional[HFDataset]]:
        """Подготовка datasets для обучения"""
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=self.config.max_length
            )
        
        # Создаем обучающий dataset
        train_dataset = HFDataset.from_dict({
            'text': train_texts,
            'labels': train_labels
        })
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        
        # Создаем валидационный dataset если предоставлен
        val_dataset = None
        if val_texts is not None and val_labels is not None:
            val_dataset = HFDataset.from_dict({
                'text': val_texts,
                'labels': val_labels
            })
            val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """Вычисление метрик для оценки модели"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, 
              train_texts: List[str], 
              train_labels: List[int],
              val_texts: Optional[List[str]] = None,
              val_labels: Optional[List[int]] = None):
        """Обучение модели"""
        
        if self.model is None:
            self.initialize_model()
        
        # Подготовка datasets
        train_dataset, val_dataset = self.prepare_datasets(
            train_texts, train_labels, val_texts, val_labels
        )
        
        # Настройка аргументов обучения
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=f'{self.config.output_dir}/logs',
            logging_steps=self.config.logging_steps,
            eval_strategy="steps" if val_dataset else "no",  # Исправлено название
            eval_steps=self.config.eval_steps if val_dataset else None,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="f1" if val_dataset else None,
            learning_rate=self.config.learning_rate,
        )
        
        # Создание тренера
        trainer_kwargs = {
            'model': self.model,
            'args': training_args,
            'train_dataset': train_dataset,
            'tokenizer': self.tokenizer,
            'compute_metrics': self.compute_metrics,
        }
        
        if val_dataset:
            trainer_kwargs['eval_dataset'] = val_dataset
            trainer_kwargs['callbacks'] = [EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)]
        
        self.trainer = Trainer(**trainer_kwargs)
        
        # Обучение
        logger.info("Начинаю обучение...")
        train_result = self.trainer.train()
        
        # Сохранение модели
        self.trainer.save_model()
        
        logger.info("Обучение завершено!")
        return train_result
    
    def predict(self, texts: List[str], return_probabilities: bool = False) -> List:
        """Предсказание для списка текстов"""
        if self.model is None:
            raise ValueError("Модель не инициализирована. Вызовите initialize_model() или train().")
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding='max_length',
                    max_length=self.config.max_length
                )
                
                outputs = self.model(**inputs)
                
                if return_probabilities:
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predictions.append(probs.cpu().numpy()[0])
                else:
                    predicted_class = torch.argmax(outputs.logits, dim=-1)
                    predictions.append(predicted_class.item())
        
        return predictions
    
    def save_model(self, path: str):
        """Сохранение модели"""
        if self.trainer:
            self.trainer.save_model(path)
        else:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        
        # Сохраняем конфигурацию
        config_path = os.path.join(path, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Модель сохранена в {path}")
    
    def load_model(self, path: str):
        """Загрузка модели"""
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Загружаем конфигурацию если есть
        config_path = os.path.join(path, "training_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                self.config = TrainingConfig(**config_dict)
        
        logger.info(f"Модель загружена из {path}")
    
    def evaluate(self, test_texts: List[str], test_labels: List[int]) -> Dict:
        """Оценка модели на тестовых данных"""
        if self.model is None:
            raise ValueError("Модель не инициализирована.")
        
        test_dataset, _ = self.prepare_datasets(test_texts, test_labels)
        
        if self.trainer is None:
            # Создаем временный trainer для оценки
            training_args = TrainingArguments(
                output_dir="./temp",
                per_device_eval_batch_size=self.config.batch_size,
            )
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics,
            )
        
        eval_result = self.trainer.evaluate(test_dataset)
        return eval_result


def create_sample_data() -> Tuple[List[str], List[int]]:
    """Создание примерных данных для демонстрации"""
    texts = [
        "Этот фильм просто великолепен! Отличная игра актеров.",
        "Ужасный фильм, потратил время зря.",
        "Неплохой фильм, можно посмотреть.",
        "Это лучший фильм, который я когда-либо видел!",
        "Скучно и предсказуемо.",
        "Отличная режиссура и сценарий.",
        "Не рекомендую к просмотру.",
        "Хороший фильм для семейного просмотра.",
        "Полная ерунда, не стоит времени.",
        "Интересный сюжет и хорошие спецэффекты."
    ]
    
    labels = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]  # 1 - положительный, 0 - отрицательный
    
    return texts, labels


if __name__ == "__main__":
    # Пример использования
    logging.basicConfig(level=logging.INFO)
    
    # Создаем конфигурацию
    config = TrainingConfig(
        num_epochs=2,
        batch_size=8,
        output_dir="./models/modernbert_sentiment"
    )
    
    # Создаем классификатор
    classifier = ModernBERTClassifier(config)
    
    # Создаем примерные данные
    texts, labels = create_sample_data()
    
    # Разделяем на обучающую и валидационную выборки
    split_idx = int(0.8 * len(texts))
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    # Обучаем модель
    classifier.train(train_texts, train_labels, val_texts, val_labels)
    
    # Тестируем предсказания
    test_texts = ["Отличный фильм!", "Ужасно скучно."]
    predictions = classifier.predict(test_texts)
    print(f"Предсказания: {predictions}") 