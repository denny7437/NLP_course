#!/usr/bin/env python3
"""
Скрипт подготовки BERT Pre-Filter модели для ONNX
Эпик 1: Модели - портирование BERT → ONNX
"""

import os
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import onnx
from optimum.onnxruntime import ORTModelForSequenceClassification

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTPreFilterTrainer:
    """Тренировка и экспорт BERT модели для Pre-Filter"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.artifacts_dir = Path("models/artifacts")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Подготовка датасета для безопасности контента"""
        logger.info("Подготовка датасета...")
        
        # Используем комбинацию датасетов для безопасности
        # В реальном проекте здесь был бы внутренний датасет
        try:
            # Попробуем загрузить существующий датасет
            dataset = load_dataset("unitary/toxic-bert", split="train[:10000]")
            
            # Преобразуем в бинарную классификацию safe/unsafe
            def process_example(example):
                # Простая эвристика: если есть любая токсичность, то unsafe
                is_toxic = any([
                    example.get('toxic', 0) > 0.5,
                    example.get('severe_toxic', 0) > 0.5,
                    example.get('obscene', 0) > 0.5,
                    example.get('threat', 0) > 0.5,
                    example.get('insult', 0) > 0.5,
                    example.get('identity_hate', 0) > 0.5
                ])
                return {
                    'text': example['comment_text'],
                    'label': 1 if is_toxic else 0  # 1 = unsafe, 0 = safe
                }
            
            dataset = dataset.map(process_example)
            
        except Exception as e:
            logger.warning(f"Не удалось загрузить внешний датасет: {e}")
            # Создаем синтетический датасет для демонстрации
            dataset = self._create_synthetic_dataset()
        
        # Разделение на train:test:val = 8:1:1
        dataset = dataset.train_test_split(test_size=0.2, stratify_by_column='label')
        test_val = dataset['test'].train_test_split(test_size=0.5, stratify_by_column='label')
        
        return dataset['train'], test_val['train'], test_val['test']
    
    def _create_synthetic_dataset(self) -> Dataset:
        """Создание синтетического датасета для демонстрации"""
        logger.info("Создание синтетического датасета...")
        
        safe_texts = [
            "Hello, how are you today?",
            "I love this weather",
            "Can you help me with my homework?",
            "What's your favorite book?",
            "I'm looking forward to the weekend",
            "This is a great movie",
            "Thank you for your help",
            "Have a wonderful day",
            "I appreciate your kindness",
            "Let's work together on this project"
        ] * 50  # 500 безопасных примеров
        
        unsafe_texts = [
            "I hate everyone",
            "You're stupid and worthless",
            "I want to hurt someone",
            "Kill yourself",
            "This is garbage content",
            "I'm going to destroy everything",
            "Nobody likes you",
            "You should die",
            "I hate this world",
            "Everyone is an idiot"
        ] * 50  # 500 небезопасных примеров
        
        texts = safe_texts + unsafe_texts
        labels = [0] * len(safe_texts) + [1] * len(unsafe_texts)
        
        return Dataset.from_dict({'text': texts, 'label': labels})
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Токенизация датасета"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors=None
            )
        
        return dataset.map(tokenize_function, batched=True)
    
    def train_model(self, train_dataset: Dataset, val_dataset: Dataset):
        """Обучение модели"""
        logger.info("Начало обучения модели...")
        
        # Инициализация модели
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            id2label={0: "safe", 1: "unsafe"},
            label2id={"safe": 0, "unsafe": 1}
        )
        
        # Токенизация датасетов
        train_dataset = self.tokenize_dataset(train_dataset)
        val_dataset = self.tokenize_dataset(val_dataset)
        
        # Настройки обучения
        training_args = TrainingArguments(
            output_dir=self.artifacts_dir / "checkpoints",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=self.artifacts_dir / "logs",
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
        )
        
        # Метрики
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted'
            )
            accuracy = accuracy_score(labels, predictions)
            return {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        
        # Тренер
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=compute_metrics,
        )
        
        # Обучение
        trainer.train()
        
        # Сохранение лучшей модели
        best_model_path = self.artifacts_dir / "bert_classifier"
        trainer.save_model(best_model_path)
        self.tokenizer.save_pretrained(best_model_path)
        
        logger.info(f"Модель сохранена в {best_model_path}")
    
    def export_to_onnx(self):
        """Экспорт в ONNX формат"""
        logger.info("Экспорт в ONNX...")
        
        model_path = self.artifacts_dir / "bert_classifier"
        onnx_path = self.artifacts_dir / "pre_filter.onnx"
        
        # Загрузка обученной модели
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Конвертация в ONNX через Optimum
        ort_model = ORTModelForSequenceClassification.from_pretrained(
            model_path, 
            from_transformers=True
        )
        
        # Сохранение ONNX модели
        ort_model.save_pretrained(self.artifacts_dir / "onnx_model")
        
        # Копируем основной файл модели
        import shutil
        onnx_model_file = self.artifacts_dir / "onnx_model" / "model.onnx"
        if onnx_model_file.exists():
            shutil.copy(onnx_model_file, onnx_path)
            logger.info(f"ONNX модель сохранена в {onnx_path}")
        
        # Оптимизация модели для инференса
        self._optimize_onnx_model(onnx_path)
    
    def _optimize_onnx_model(self, onnx_path: Path):
        """Оптимизация ONNX модели"""
        try:
            from onnxruntime.tools import optimizer
            
            # Оптимизация для CPU инференса
            optimized_path = onnx_path.with_suffix('.optimized.onnx')
            
            # Базовая оптимизация
            optimizer.optimize_model(
                str(onnx_path),
                str(optimized_path),
                file_type='onnx',
                optimization_level='basic'
            )
            
            # Заменяем исходный файл оптимизированным
            optimized_path.replace(onnx_path)
            logger.info("ONNX модель оптимизирована")
            
        except ImportError:
            logger.warning("onnxruntime.tools недоступен, пропускаем оптимизацию")
    
    def validate_onnx_model(self, test_dataset: Dataset):
        """Валидация ONNX модели"""
        logger.info("Валидация ONNX модели...")
        
        try:
            import onnxruntime as ort
            
            onnx_path = self.artifacts_dir / "pre_filter.onnx"
            
            # Создание сессии ONNX Runtime
            session = ort.InferenceSession(str(onnx_path))
            
            # Тестирование на нескольких примерах
            test_texts = test_dataset['text'][:5]
            
            for text in test_texts:
                # Токенизация
                inputs = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="np"
                )
                
                # Инференс
                outputs = session.run(
                    None,
                    {
                        'input_ids': inputs['input_ids'],
                        'attention_mask': inputs['attention_mask']
                    }
                )
                
                # Предсказание
                logits = outputs[0]
                prediction = np.argmax(logits, axis=1)[0]
                confidence = float(np.max(torch.softmax(torch.tensor(logits), dim=1).numpy()))
                
                status = "unsafe" if prediction == 1 else "safe"
                logger.info(f"Text: {text[:50]}... -> {status} (confidence: {confidence:.3f})")
                
            logger.info("Валидация ONNX модели успешна")
            
        except Exception as e:
            logger.error(f"Ошибка валидации ONNX: {e}")
    
    def run_full_pipeline(self):
        """Полный пайплайн обучения и экспорта"""
        logger.info("Запуск полного пайплайна...")
        
        # 1. Подготовка данных
        train_ds, val_ds, test_ds = self.prepare_dataset()
        logger.info(f"Размеры датасетов - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
        
        # 2. Обучение модели
        self.train_model(train_ds, val_ds)
        
        # 3. Экспорт в ONNX
        self.export_to_onnx()
        
        # 4. Валидация
        self.validate_onnx_model(test_ds)
        
        logger.info("Пайплайн завершен успешно!")


if __name__ == "__main__":
    trainer = BERTPreFilterTrainer()
    trainer.run_full_pipeline() 