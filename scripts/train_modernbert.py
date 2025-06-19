#!/usr/bin/env python3
"""
Скрипт для обучения ModernBERT классификатора
Использование:
python scripts/train_modernbert.py --config-file config.json
или 
python scripts/train_modernbert.py --quick-demo
"""

import argparse
import json
import sys
import os
import logging
from pathlib import Path

# Добавляем корневую папку в path
sys.path.append(str(Path(__file__).parent.parent))

from models.modernbert_classifier import ModernBERTClassifier, TrainingConfig, create_sample_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data_from_file(file_path: str):
    """Загрузка данных из файла"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = data.get('texts', [])
    labels = data.get('labels', [])
    
    if len(texts) != len(labels):
        raise ValueError("Количество текстов и меток должно совпадать")
    
    return texts, labels


def create_config_from_args(args):
    """Создание конфигурации из аргументов командной строки"""
    config = TrainingConfig()
    
    if args.model_name:
        config.model_name = args.model_name
    if args.num_labels:
        config.num_labels = args.num_labels
    if args.max_length:
        config.max_length = args.max_length
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.output_dir:
        config.output_dir = args.output_dir
    
    return config


def run_quick_demo():
    """Запуск быстрого демо с примерными данными"""
    logger.info("🚀 Запускаю быстрое демо ModernBERT классификатора")
    
    # Создаем конфигурацию для демо
    config = TrainingConfig(
        model_name="answerdotai/ModernBERT-base",
        num_epochs=1,  # Быстрое обучение для демо
        batch_size=2,  # Очень маленький batch size для безопасности
        max_length=512,  # Ограничиваем длину для демо
        output_dir="./models/modernbert_demo",
        learning_rate=5e-5
    )
    
    # Создаем классификатор
    classifier = ModernBERTClassifier(config)
    
    # Создаем демо данные
    texts, labels = create_sample_data()
    logger.info(f"Создано {len(texts)} примеров для обучения")
    
    # Разделяем данные
    split_idx = int(0.7 * len(texts))
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    logger.info(f"Обучающая выборка: {len(train_texts)} примеров")
    logger.info(f"Валидационная выборка: {len(val_texts)} примеров")
    
    try:
        # Обучаем модель
        train_result = classifier.train(train_texts, train_labels, val_texts, val_labels)
        
        # Тестируем модель
        test_texts = [
            "Отличный фильм, всем рекомендую!",
            "Ужасное кино, не тратьте время.",
            "Неплохая картина, можно посмотреть",
            "Это шедевр кинематографа!"
        ]
        
        logger.info("🔮 Тестирую модель на новых данных:")
        predictions = classifier.predict(test_texts)
        probabilities = classifier.predict(test_texts, return_probabilities=True)
        
        for i, (text, pred, probs) in enumerate(zip(test_texts, predictions, probabilities)):
            sentiment = "Положительный" if pred == 1 else "Отрицательный"
            confidence = max(probs) * 100
            logger.info(f"  {i+1}. '{text}' -> {sentiment} (уверенность: {confidence:.1f}%)")
        
        # Сохраняем модель
        classifier.save_model(config.output_dir)
        logger.info(f"✅ Модель сохранена в {config.output_dir}")
        
        return classifier
        
    except Exception as e:
        logger.error(f"❌ Ошибка во время обучения: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Обучение ModernBERT классификатора")
    
    parser.add_argument("--quick-demo", action="store_true", 
                       help="Запустить быстрое демо с примерными данными")
    
    parser.add_argument("--config-file", type=str,
                       help="Путь к JSON файлу с конфигурацией")
    
    parser.add_argument("--train-data", type=str,
                       help="Путь к JSON файлу с обучающими данными")
    
    parser.add_argument("--val-data", type=str,
                       help="Путь к JSON файлу с валидационными данными")
    
    # Параметры модели
    parser.add_argument("--model-name", type=str, default="answerdotai/ModernBERT-base",
                       help="Название ModernBERT модели")
    
    parser.add_argument("--num-labels", type=int, default=2,
                       help="Количество классов для классификации")
    
    parser.add_argument("--max-length", type=int, default=512,
                       help="Максимальная длина токенизации")
    
    # Параметры обучения
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Размер батча")
    
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Скорость обучения")
    
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Количество эпох обучения")
    
    parser.add_argument("--output-dir", type=str, default="./models/modernbert_classifier",
                       help="Папка для сохранения модели")
    
    args = parser.parse_args()
    
    if args.quick_demo:
        return run_quick_demo()
    
    # Загружаем конфигурацию
    if args.config_file:
        with open(args.config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config = TrainingConfig(**config_dict)
    else:
        config = create_config_from_args(args)
    
    # Создаем классификатор
    classifier = ModernBERTClassifier(config)
    
    # Загружаем данные
    if args.train_data:
        train_texts, train_labels = load_data_from_file(args.train_data)
        logger.info(f"Загружено {len(train_texts)} обучающих примеров")
    else:
        logger.error("Необходимо указать --train-data или использовать --quick-demo")
        return
    
    val_texts, val_labels = None, None
    if args.val_data:
        val_texts, val_labels = load_data_from_file(args.val_data)
        logger.info(f"Загружено {len(val_texts)} валидационных примеров")
    
    # Обучаем модель
    logger.info("🚀 Начинаю обучение ModernBERT классификатора")
    try:
        train_result = classifier.train(train_texts, train_labels, val_texts, val_labels)
        
        # Сохраняем модель
        classifier.save_model(config.output_dir)
        logger.info(f"✅ Обучение завершено! Модель сохранена в {config.output_dir}")
        
        return classifier
        
    except Exception as e:
        logger.error(f"❌ Ошибка во время обучения: {e}")
        raise


if __name__ == "__main__":
    main() 