#!/usr/bin/env python3
"""
Улучшенное обучение BERT детектора оскорбления чувств верующих
Оптимизированная версия с лучшими гиперпараметрами
"""

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback
)
import os
from torch.utils.data import Dataset
import logging
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OptimizedReligiousDataset(Dataset):
    """Оптимизированный датасет для религиозного контента"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Токенизация с оптимизированными параметрами
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


def analyze_dataset(df):
    """Детальный анализ датасета"""
    logger.info("📊 АНАЛИЗ ДАТАСЕТА")
    logger.info("=" * 50)
    
    # Основная статистика
    total_samples = len(df)
    logger.info(f"📈 Общее количество примеров: {total_samples}")
    
    # Распределение классов
    class_counts = df['label'].value_counts().sort_index()
    logger.info(f"📊 Распределение классов:")
    for label, count in class_counts.items():
        label_name = "Безопасный" if label == 0 else "Опасный"
        percentage = (count / total_samples) * 100
        logger.info(f"   {label_name} (label={label}): {count} ({percentage:.1f}%)")
    
    # Статистика длины текстов
    text_lengths = df['text'].str.len()
    word_counts = df['text'].str.split().str.len()
    
    logger.info(f"\n📝 Статистика текстов:")
    logger.info(f"   Средняя длина символов: {text_lengths.mean():.1f}")
    logger.info(f"   Медианная длина символов: {text_lengths.median():.1f}")
    logger.info(f"   Максимальная длина: {text_lengths.max()}")
    logger.info(f"   Среднее количество слов: {word_counts.mean():.1f}")
    logger.info(f"   Медианное количество слов: {word_counts.median():.1f}")
    
    # Проверка баланса классов
    balance_ratio = min(class_counts) / max(class_counts)
    logger.info(f"\n⚖️ Баланс классов: {balance_ratio:.3f}")
    if balance_ratio < 0.5:
        logger.warning("⚠️ Датасет несбалансирован! Рекомендуется использовать взвешенную функцию потерь")
    
    return {
        'total_samples': total_samples,
        'class_distribution': class_counts,
        'balance_ratio': balance_ratio,
        'avg_length': text_lengths.mean(),
        'avg_words': word_counts.mean()
    }


class OptimizedTrainer(Trainer):
    """Оптимизированный тренер с дополнительными метриками"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Взвешенная функция потерь для несбалансированных данных"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # Взвешенная кросс-энтропия
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5], device=labels.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """Расширенные метрики для оценки"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Базовые метрики
    accuracy = accuracy_score(labels, predictions)
    
    # Детальный отчет
    report = classification_report(
        labels, predictions,
        target_names=['Безопасный', 'Опасный'],
        output_dict=True
    )
    
    # Матрица ошибок
    cm = confusion_matrix(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1_safe': report['Безопасный']['f1-score'],
        'f1_dangerous': report['Опасный']['f1-score'],
        'precision_safe': report['Безопасный']['precision'],
        'precision_dangerous': report['Опасный']['precision'],
        'recall_safe': report['Безопасный']['recall'],
        'recall_dangerous': report['Опасный']['recall'],
        'macro_f1': report['macro avg']['f1-score']
    }


def train_optimized_model():
    """Обучение оптимизированной модели"""
    logger.info("🚀 ЗАПУСК ОПТИМИЗИРОВАННОГО ОБУЧЕНИЯ BERT")
    logger.info("=" * 60)
    
    # Загрузка данных
    logger.info("📂 Загружаю тренировочные данные...")
    df = pd.read_csv("train - Лист1.csv")
    logger.info(f"✅ Загружено {len(df)} примеров")
    
    # Анализ датасета
    dataset_stats = analyze_dataset(df)
    
    # Подготовка данных
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    # Выбор оптимальной модели
    model_name = "cointegrated/rubert-tiny2"  # Быстрая и эффективная русская BERT
    logger.info(f"🤖 Используемая модель: {model_name}")
    
    # Инициализация токенизатора и модели
    logger.info("🔧 Инициализация модели и токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # Создание датасета
    max_length = min(256, int(dataset_stats['avg_length'] * 1.5))  # Оптимальная длина
    logger.info(f"📏 Максимальная длина последовательности: {max_length}")
    
    train_dataset = OptimizedReligiousDataset(texts, labels, tokenizer, max_length)
    
    # Создание коллатора данных
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Оптимизированные параметры обучения
    training_args = TrainingArguments(
        output_dir='./optimized_religious_classifier',
        
        # Основные параметры
        num_train_epochs=5,  # Больше эпох для лучшего обучения
        per_device_train_batch_size=16,  # Оптимальный размер батча
        per_device_eval_batch_size=32,
        
        # Оптимизация обучения
        learning_rate=3e-5,  # Оптимальный learning rate для BERT
        weight_decay=0.01,
        warmup_steps=200,  # Разогрев для стабильности
        
        # Логирование и сохранение
        logging_steps=50,
        save_steps=200,
        save_strategy="steps",
        save_total_limit=3,
        
        # Оценка (используем весь датасет для обучения)
        eval_strategy="no",  # Отключаем валидацию
        
        # Оптимизация производительности
        dataloader_num_workers=4,
        remove_unused_columns=False,
        
        # Стабилизация обучения
        seed=42,
        fp16=False,  # Отключаем для стабильности на CPU
        
        # Дополнительные параметры
        push_to_hub=False,
        report_to=None,  # Отключаем wandb/tensorboard
        
        # Градиентная оптимизация
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        
        # Расписание learning rate
        lr_scheduler_type="cosine",
        
        # Использование CPU
        use_cpu=True,
    )
    
    # Создание оптимизированного тренера
    trainer = OptimizedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Обучение
    logger.info("🎯 НАЧИНАЮ ОПТИМИЗИРОВАННОЕ ОБУЧЕНИЕ...")
    logger.info(f"   📊 Размер обучающей выборки: {len(train_dataset)}")
    logger.info(f"   🔢 Количество эпох: {training_args.num_train_epochs}")
    logger.info(f"   📦 Размер батча: {training_args.per_device_train_batch_size}")
    logger.info(f"   📈 Learning rate: {training_args.learning_rate}")
    
    # Запуск обучения
    train_result = trainer.train()
    
    # Логирование результатов обучения
    logger.info("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    logger.info(f"📊 Финальная функция потерь: {train_result.training_loss:.4f}")
    logger.info(f"⏱️ Время обучения: {train_result.metrics['train_runtime']:.2f} секунд")
    logger.info(f"🔄 Общее количество шагов: {train_result.metrics['train_steps_per_second']:.2f} шагов/сек")
    
    # Сохранение модели
    logger.info("💾 Сохраняю оптимизированную модель...")
    trainer.save_model()
    tokenizer.save_pretrained('./optimized_religious_classifier')
    
    # Тестирование на тренировочных данных
    logger.info("🧪 Тестирую модель на тренировочных данных...")
    
    # Предсказания на всем датасете
    train_predictions = trainer.predict(train_dataset)
    predicted_labels = np.argmax(train_predictions.predictions, axis=1)
    
    # Детальные метрики
    accuracy = accuracy_score(labels, predicted_labels)
    report = classification_report(
        labels, predicted_labels,
        target_names=['Безопасный', 'Опасный'],
        output_dict=True
    )
    cm = confusion_matrix(labels, predicted_labels)
    
    # Вывод результатов
    logger.info("\n🎯 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ НА ТРЕНИРОВОЧНЫХ ДАННЫХ:")
    logger.info("=" * 55)
    logger.info(f"🎯 Точность: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    logger.info(f"\n📊 Детальные метрики:")
    logger.info(f"   Безопасный контент:")
    logger.info(f"     Precision: {report['Безопасный']['precision']:.4f}")
    logger.info(f"     Recall: {report['Безопасный']['recall']:.4f}")
    logger.info(f"     F1-score: {report['Безопасный']['f1-score']:.4f}")
    
    logger.info(f"   Опасный контент:")
    logger.info(f"     Precision: {report['Опасный']['precision']:.4f}")
    logger.info(f"     Recall: {report['Опасный']['recall']:.4f}")
    logger.info(f"     F1-score: {report['Опасный']['f1-score']:.4f}")
    
    logger.info(f"\n📈 Общие метрики:")
    logger.info(f"   Macro F1-score: {report['macro avg']['f1-score']:.4f}")
    logger.info(f"   Weighted F1-score: {report['weighted avg']['f1-score']:.4f}")
    
    # Матрица ошибок
    logger.info(f"\n🔍 Матрица ошибок:")
    logger.info(f"                Предсказано")
    logger.info(f"            Безопасный  Опасный")
    logger.info(f"Истинно")
    logger.info(f"Безопасный      {cm[0,0]:4d}     {cm[0,1]:4d}")
    logger.info(f"Опасный         {cm[1,0]:4d}     {cm[1,1]:4d}")
    
    # Анализ ошибок
    false_positives = cm[0,1]  # Безопасные, классифицированные как опасные
    false_negatives = cm[1,0]  # Опасные, классифицированные как безопасные
    
    logger.info(f"\n❌ Анализ ошибок:")
    logger.info(f"   Ложные срабатывания (FP): {false_positives}")
    logger.info(f"   Пропущенные опасные (FN): {false_negatives}")
    
    if false_positives > false_negatives:
        logger.info("   📝 Модель склонна к переклассификации (слишком осторожна)")
    elif false_negatives > false_positives:
        logger.info("   📝 Модель пропускает опасный контент (недостаточно чувствительна)")
    else:
        logger.info("   📝 Сбалансированное распределение ошибок")
    
    # Тестирование на примерах
    logger.info("\n🧪 ТЕСТИРОВАНИЕ НА НОВЫХ ПРИМЕРАХ:")
    logger.info("=" * 45)
    
    test_examples = [
        ("Молитва помогает найти внутренний покой и гармонию", 0),
        ("Религиозные традиции объединяют людей в сообщества", 0),
        ("Духовные практики способствуют личностному росту", 0),
        ("Все верующие - дураки и фанатики без мозгов", 1),
        ("Церковь должна быть уничтожена огнем и мечом", 1),
        ("Религия - это опиум для народа и инструмент угнетения", 1),
        ("Вера в бога - признак психического расстройства", 1),
        ("Различные конфессии имеют право на мирное существование", 0),
    ]
    
    # Переводим модель в режим предсказания
    model.eval()
    device = torch.device('cpu')
    model.to(device)
    
    correct_predictions = 0
    for i, (text, expected) in enumerate(test_examples, 1):
        # Токенизация
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=max_length
        )
        
        # Предсказание
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
        
        # Анализ результата
        expected_name = "Безопасный" if expected == 0 else "Опасный"
        predicted_name = "Безопасный" if predicted_class == 0 else "Опасный"
        correct = "✅" if predicted_class == expected else "❌"
        
        if predicted_class == expected:
            correct_predictions += 1
        
        logger.info(f"  {i}. {correct} '{text[:60]}{'...' if len(text) > 60 else ''}'")
        logger.info(f"     Ожидалось: {expected_name}, Получено: {predicted_name}")
    
    test_accuracy = correct_predictions / len(test_examples)
    logger.info(f"\n📊 Точность на тестовых примерах: {test_accuracy:.2%} ({correct_predictions}/{len(test_examples)})")
    
    # Итоговая оценка
    logger.info("\n🏆 ИТОГОВАЯ ОЦЕНКА МОДЕЛИ:")
    logger.info("=" * 35)
    
    if accuracy >= 0.95:
        logger.info("🌟 ОТЛИЧНО! Модель показывает превосходные результаты")
    elif accuracy >= 0.90:
        logger.info("🎯 ХОРОШО! Модель показывает хорошие результаты")
    elif accuracy >= 0.80:
        logger.info("📈 УДОВЛЕТВОРИТЕЛЬНО. Есть потенциал для улучшения")
    else:
        logger.info("⚠️ ТРЕБУЕТ ДОРАБОТКИ. Низкая точность")
    
    # Рекомендации
    logger.info(f"\n💡 РЕКОМЕНДАЦИИ:")
    if false_positives > 50:
        logger.info("   - Рассмотреть снижение чувствительности модели")
        logger.info("   - Добавить больше разнообразных безопасных примеров")
    
    if false_negatives > 20:
        logger.info("   - Увеличить количество опасных примеров в обучении")
        logger.info("   - Рассмотреть увеличение веса класса 'опасный'")
    
    if accuracy < 0.90:
        logger.info("   - Увеличить количество эпох обучения")
        logger.info("   - Попробовать другую архитектуру модели")
        logger.info("   - Улучшить качество данных")
    
    logger.info("\n🎉 ОБУЧЕНИЕ ОПТИМИЗИРОВАННОЙ МОДЕЛИ ЗАВЕРШЕНО!")
    logger.info("✨ Модель сохранена в папке ./optimized_religious_classifier")
    
    return {
        'accuracy': accuracy,
        'model_path': './optimized_religious_classifier',
        'training_loss': train_result.training_loss,
        'metrics': report
    }


if __name__ == "__main__":
    try:
        results = train_optimized_model()
        logger.info(f"🎯 Финальная точность: {results['accuracy']:.4f}")
    except Exception as e:
        logger.error(f"❌ Ошибка при обучении: {e}")
        raise 