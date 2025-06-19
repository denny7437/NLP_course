#!/usr/bin/env python3
"""
Обучение BERT для детекции оскорбления чувств верующих
(альтернатива ModernBERT для избежания ошибок с DTensor)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReligiousContentDetector:
    """Детектор религиозного контента на основе BERT"""
    
    def __init__(self):
        self.model_name = "DeepPavlov/rubert-base-cased"  # Более мощная модель
        self.max_length = 256  # Оптимальная длина для GPU
        self.tokenizer = None
        self.model = None
        self.trainer = None
    
    def initialize_model(self):
        """Инициализация модели"""
        logger.info(f"Загружаю модель {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )
        
        # Используем GPU если доступен
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        logger.info(f"🖥️ Используется устройство: {device}")
        
        logger.info("Модель загружена")
    
    def prepare_dataset(self, texts, labels):
        """Подготовка датасета"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
        
        dataset = Dataset.from_dict({
            'text': texts,
            'labels': labels
        })
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def compute_metrics(self, eval_pred):
        """Вычисление расширенных метрик"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions, output_dict=True, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1_macro': report['macro avg']['f1-score'],
            'f1_weighted': report['weighted avg']['f1-score'],
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall']
        }
    
    def train(self, train_texts, train_labels, val_texts, val_labels, epochs=5):
        """Обучение модели"""
        if self.model is None:
            self.initialize_model()
        
        # Подготовка датасетов
        train_dataset = self.prepare_dataset(train_texts, train_labels)
        val_dataset = self.prepare_dataset(val_texts, val_labels)
        
        # Вычисляем класс-весы для несбалансированных данных
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
        logger.info(f"🔢 Веса классов: {class_weights_dict}")
        
        # Настройка обучения (улучшенные параметры)
        training_args = TrainingArguments(
            output_dir='./bert_religious_classifier_improved',
            num_train_epochs=epochs,
            per_device_train_batch_size=8,  # Уменьшим для стабильности
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2,  # Эмулируем batch_size=16
            warmup_steps=100,  # Больше warmup шагов
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="epoch",  # Оценка каждую эпоху
            save_strategy="no",  # Отключаем автосохранение
            learning_rate=2e-5,  # Меньший learning rate
            lr_scheduler_type="cosine",  # Cosine scheduler
            load_best_model_at_end=False,  # Отключаем загрузку лучшей модели
            fp16=torch.cuda.is_available(),  # Mixed precision если GPU
            dataloader_num_workers=0,
            report_to=None,  # Отключаем wandb
            seed=42,  # Фиксируем seed для воспроизводимости
        )
        
        # Кастомный класс для взвешенной функции потерь
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get('logits')
                
                # Взвешенная функция потерь
                loss_fct = torch.nn.CrossEntropyLoss(
                    weight=torch.tensor([class_weights_dict[0], class_weights_dict[1]], 
                                      dtype=torch.float32, device=logits.device)
                )
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss
        
        # Создание тренера с взвешенной функцией потерь
        self.trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        # Обучение
        logger.info("Начинаю обучение...")
        try:
            train_result = self.trainer.train()
            logger.info("Обучение завершено!")
            
            # Сохраняем модель (с исправлением contiguous tensor)
            logger.info("💾 Сохраняю улучшенную модель...")
            
            # Делаем все тензоры contiguous перед сохранением
            for name, param in self.model.named_parameters():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()
            
            self.trainer.save_model()
            self.tokenizer.save_pretrained('./bert_religious_classifier_improved')
            logger.info("✅ Улучшенная модель сохранена")
            
            return train_result
        except Exception as e:
            logger.error(f"❌ Ошибка при обучении: {e}")
            return None
    
    def predict(self, texts):
        """Предсказание с использованием GPU"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length
                )
                
                # Переносим входные данные на устройство
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                predicted_class = torch.argmax(outputs.logits, dim=-1)
                predictions.append(predicted_class.item())
        
        return predictions
    
    def evaluate_on_test(self, test_texts, test_labels):
        """Оценка на тестовых данных"""
        predictions = self.predict(test_texts)
        
        accuracy = accuracy_score(test_labels, predictions)
        report = classification_report(
            test_labels, 
            predictions, 
            target_names=['Безопасный', 'Опасный'],
            output_dict=True
        )
        
        return accuracy, predictions, report


def main():
    """Главная функция"""
    logger.info("🎯 Обучение улучшенного BERT детектора оскорбления чувств верующих")
    
    # Проверяем доступность GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"🚀 GPU доступен: {gpu_name}")
        logger.info(f"💾 Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.info("🖥️ GPU недоступен, будет использоваться CPU")
    
    # Загрузка данных
    logger.info("📊 Загружаю данные...")
    df = pd.read_csv("train - Лист1.csv")
    logger.info(f"Загружено {len(df)} примеров")
    
    # Анализ данных
    class_distribution = df['label'].value_counts().sort_index()
    logger.info("📈 Распределение классов:")
    logger.info(f"  Класс 0 (безопасный): {class_distribution[0]} ({class_distribution[0]/len(df)*100:.1f}%)")
    logger.info(f"  Класс 1 (опасный): {class_distribution[1]} ({class_distribution[1]/len(df)*100:.1f}%)")
    
    # Подготовка данных
    X = df['text'].tolist()
    y = df['label'].tolist()
    
    # Разделение данных
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Тренировочная выборка: {len(X_train)}")
    logger.info(f"Валидационная выборка: {len(X_val)}")
    logger.info(f"Тестовая выборка: {len(X_test)}")
    
    # Создание и обучение детектора
    detector = ReligiousContentDetector()
    
    # Обучение с улучшенными параметрами
    train_result = detector.train(X_train, y_train, X_val, y_val, epochs=5)
    
    if train_result is not None:
        # Оценка на тестовых данных
        logger.info("📊 Оцениваю модель на тестовых данных...")
        accuracy, predictions, report = detector.evaluate_on_test(X_test, y_test)
        
        logger.info(f"\n🎯 Результаты:")
        logger.info(f"Точность: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        logger.info(f"\nДетальные метрики:")
        logger.info(f"Безопасный контент:")
        logger.info(f"  Precision: {report['Безопасный']['precision']:.3f}")
        logger.info(f"  Recall: {report['Безопасный']['recall']:.3f}")
        logger.info(f"  F1-score: {report['Безопасный']['f1-score']:.3f}")
        
        logger.info(f"Опасный контент:")
        logger.info(f"  Precision: {report['Опасный']['precision']:.3f}")
        logger.info(f"  Recall: {report['Опасный']['recall']:.3f}")
        logger.info(f"  F1-score: {report['Опасный']['f1-score']:.3f}")
        
        # Матрица ошибок
        cm = confusion_matrix(y_test, predictions)
        logger.info(f"\n🔍 Матрица ошибок:")
        logger.info(f"             Предсказано")
        logger.info(f"         Безопасный  Опасный")
        logger.info(f"Истинно")
        logger.info(f"Безопасный    {cm[0,0]:3d}      {cm[0,1]:3d}")
        logger.info(f"Опасный       {cm[1,0]:3d}      {cm[1,1]:3d}")
        
        logger.info("\n🎉 Обучение успешно завершено!")
        logger.info("✨ BERT обучен для детекции оскорбления чувств верующих")
    else:
        logger.error("❌ Обучение не удалось")


if __name__ == "__main__":
    main() 