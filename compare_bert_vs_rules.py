#!/usr/bin/env python3
"""
Сравнение обученной BERT модели с rule-based подходом
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BERTDetector:
    """Оптимизированная BERT модель"""
    
    def __init__(self, model_path="./bert_religious_classifier_improved"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.max_length = 256
        
    def load_model(self):
        """Загрузка оптимизированной модели"""
        logger.info(f"🔄 Загружаю оптимизированную BERT модель из {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        
        # Переводим в режим предсказания
        self.model.eval()
        
        logger.info("✅ Оптимизированная BERT модель загружена")
    
    def predict(self, texts):
        """Предсказание"""
        if self.model is None:
            self.load_model()
        
        predictions = []
        # Используем GPU если доступен, иначе CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️ Используется устройство: {device}")
        self.model.to(device)
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length
                )
                
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                predicted_class = torch.argmax(outputs.logits, dim=-1)
                predictions.append(predicted_class.item())
        
        return predictions


class RuleBasedDetector:
    """Rule-based детектор (улучшенный)"""
    
    def __init__(self):
        # Те же паттерны, что используются для автоматической разметки
        self.dangerous_patterns = [
            # Прямые оскорбления
            'дураки', 'идиоты', 'тупые', 'глупые', 'слабоумные', 'фанатики',
            'рабы', 'зомби', 'отсталые', 'примитивные', 'варвары', 'дикари',
            
            # Призывы к насилию
            'уничтожить', 'сжечь', 'убить', 'смерть', 'сожжет', 'выжечь',
            'запрещена', 'должна быть запрещена',
            
            # Обвинения в обмане
            'ложь', 'обман', 'мошенничество', 'лживая', 'сказки', 'театр',
            'лицемерие', 'лицемерный', 'прикрытие',
            
            # Сектантская терминология
            'секта', 'сектанты', 'промывание мозгов', 'зомбируют', 'культ',
            'тоталитарная секта', 'рабоверующих',
            
            # Радикальные высказывания
            'чума', 'вирус', 'отравляет', 'раковая опухоль', 'паразит',
            'психическое расстройство', 'калечащий психику',
            
            # Антирелигиозная агрессия
            'выздоровеет, только когда', 'последнюю церковь', 'последнюю библию'
        ]
    
    def predict(self, texts):
        """Предсказание на основе правил"""
        predictions = []
        
        for text in texts:
            text_lower = text.lower()
            
            # Проверяем наличие опасных паттернов
            is_dangerous = any(pattern in text_lower for pattern in self.dangerous_patterns)
            
            if is_dangerous:
                predictions.append(1)  # Опасный
            else:
                predictions.append(0)  # Безопасный
        
        return predictions


def evaluate_model(model, test_texts, test_labels, model_name):
    """Оценка модели"""
    logger.info(f"🔍 Тестирую {model_name}...")
    
    start_time = time.time()
    predictions = model.predict(test_texts)
    end_time = time.time()
    
    prediction_time = end_time - start_time
    logger.info(f"⏱️ Время предсказаний: {prediction_time:.3f} секунд")
    logger.info(f"🔢 Получено {len(predictions)} предсказаний")
    
    # Вычисляем метрики
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(
        test_labels,
        predictions,
        target_names=['Безопасный', 'Опасный'],
        output_dict=True
    )
    
    cm = confusion_matrix(test_labels, predictions)
    
    return {
        'predictions': predictions,
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'prediction_time': prediction_time
    }


def print_results(results, model_name):
    """Вывод результатов"""
    logger.info(f"\n🎯 {model_name}:")
    logger.info(f"   Точность: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
    logger.info(f"   Время предсказаний: {results['prediction_time']:.3f} сек")
    
    report = results['report']
    logger.info(f"   Безопасный контент:")
    logger.info(f"     Precision: {report['Безопасный']['precision']:.3f}")
    logger.info(f"     Recall: {report['Безопасный']['recall']:.3f}")
    logger.info(f"     F1-score: {report['Безопасный']['f1-score']:.3f}")
    
    logger.info(f"   Опасный контент:")
    logger.info(f"     Precision: {report['Опасный']['precision']:.3f}")
    logger.info(f"     Recall: {report['Опасный']['recall']:.3f}")
    logger.info(f"     F1-score: {report['Опасный']['f1-score']:.3f}")
    
    cm = results['confusion_matrix']
    logger.info(f"   Матрица ошибок:")
    logger.info(f"              Предсказано")
    logger.info(f"          Безопасный  Опасный")
    logger.info(f"   Истинно")
    logger.info(f"   Безопасный    {cm[0,0]:3d}       {cm[0,1]:3d}")
    logger.info(f"   Опасный       {cm[1,0]:3d}       {cm[1,1]:3d}")


def create_labels(texts):
    """Создание автоматической разметки для тестовых данных"""
    logger.info("🏷️ Создаю автоматическую разметку...")
    
    # Ключевые слова для опасного контента
    dangerous_patterns = [
        # Прямые оскорбления
        'дураки', 'идиоты', 'тупые', 'глупые', 'слабоумные', 'фанатики',
        'рабы', 'зомби', 'отсталые', 'примитивные', 'варвары', 'дикари',
        
        # Призывы к насилию
        'уничтожить', 'сжечь', 'убить', 'смерть', 'сожжет', 'выжечь',
        'запрещена', 'должна быть запрещена',
        
        # Обвинения в обмане
        'ложь', 'обман', 'мошенничество', 'лживая', 'сказки', 'театр',
        'лицемерие', 'лицемерный', 'прикрытие',
        
        # Сектантская терминология
        'секта', 'сектанты', 'промывание мозгов', 'зомбируют', 'культ',
        'тоталитарная секта', 'рабоверующих',
        
        # Радикальные высказывания
        'чума', 'вирус', 'отравляет', 'раковая опухоль', 'паразит',
        'психическое расстройство', 'калечащий психику',
        
        # Антирелигиозная агрессия
        'выздоровеет, только когда', 'последнюю церковь', 'последнюю библию'
    ]
    
    labels = []
    dangerous_count = 0
    
    for text in texts:
        text_lower = text.lower()
        
        # Проверяем наличие опасных паттернов
        is_dangerous = any(pattern in text_lower for pattern in dangerous_patterns)
        
        if is_dangerous:
            labels.append(1)  # Опасный
            dangerous_count += 1
        else:
            labels.append(0)  # Безопасный
    
    safe_count = len(texts) - dangerous_count
    logger.info(f"📊 Автоматическая разметка:")
    logger.info(f"   ✅ Безопасных: {safe_count} ({safe_count/len(texts)*100:.1f}%)")
    logger.info(f"   🚨 Опасных: {dangerous_count} ({dangerous_count/len(texts)*100:.1f}%)")
    
    return labels


def main():
    """Главная функция"""
    logger.info("🎯 СРАВНЕНИЕ ОПТИМИЗИРОВАННОЙ BERT МОДЕЛИ С RULE-BASED ПОДХОДОМ")
    logger.info("=" * 65)
    
    # Проверяем доступность GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"🚀 GPU доступен: {gpu_name}")
        logger.info(f"💾 Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.info("🖥️ GPU недоступен, будет использоваться CPU")
    
    logger.info("📂 Загружаю исходные тестовые данные...")
    
    # Загрузка данных
    df = pd.read_csv("eval_modernbert - Лист1.csv")
    logger.info(f"✅ Загружено {len(df)} тестовых примеров")
    
    test_texts = df['text'].tolist()
    
    # Создаем автоматическую разметку
    test_labels = create_labels(test_texts)
    
    # Сохраняем размеченные данные
    labeled_df = pd.DataFrame({
        'text': test_texts,
        'label': test_labels
    })
    labeled_df.to_csv("eval_modernbert_auto_labeled.csv", index=False)
    logger.info("💾 Размеченные данные сохранены в eval_modernbert_auto_labeled.csv")
    
    # Анализ данных
    logger.info("\n📊 АНАЛИЗ ТЕСТОВОГО ДАТАСЕТА")
    logger.info("=" * 35)
    logger.info(f"📈 Общее количество примеров: {len(test_texts)}")
    
    safe_count = sum(1 for label in test_labels if label == 0)
    dangerous_count = sum(1 for label in test_labels if label == 1)
    
    logger.info(f"✅ Безопасных (label=0): {safe_count} ({safe_count/len(test_labels)*100:.1f}%)")
    logger.info(f"🚨 Опасных (label=1): {dangerous_count} ({dangerous_count/len(test_labels)*100:.1f}%)")
    
    # Создание моделей
    logger.info("\n🆚 СРАВНЕНИЕ МОДЕЛЕЙ")
    logger.info("=" * 25)
    
    bert_model = BERTDetector()
    rule_model = RuleBasedDetector()
    
    # Тестирование моделей
    bert_results = evaluate_model(bert_model, test_texts, test_labels, "Оптимизированная BERT модель")
    rule_results = evaluate_model(rule_model, test_texts, test_labels, "Rule-based модель")
    
    # Вывод результатов
    logger.info("\n📊 РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    logger.info("=" * 30)
    
    print_results(bert_results, "Оптимизированная BERT модель")
    print_results(rule_results, "Rule-based модель")
    
    # Сравнение согласованности
    bert_preds = bert_results['predictions']
    rule_preds = rule_results['predictions']
    
    agreement = sum(1 for i in range(len(bert_preds)) if bert_preds[i] == rule_preds[i])
    agreement_rate = agreement / len(bert_preds)
    
    logger.info(f"\n🤝 Согласованность моделей:")
    logger.info(f"   Совпадающие предсказания: {agreement}/{len(bert_preds)} ({agreement_rate*100:.1f}%)")
    
    # Анализ расхождений
    disagreements = []
    for i in range(len(bert_preds)):
        if bert_preds[i] != rule_preds[i]:
            disagreements.append((i, test_texts[i], bert_preds[i], rule_preds[i], test_labels[i]))
    
    if disagreements:
        logger.info(f"\n🔍 Примеры расхождений (первые 5):")
        for idx, (i, text, bert_pred, rule_pred, true_label) in enumerate(disagreements[:5], 1):
            bert_name = "Безопасный" if bert_pred == 0 else "Опасный"
            rule_name = "Безопасный" if rule_pred == 0 else "Опасный"
            true_name = "Безопасный" if true_label == 0 else "Опасный"
            
            logger.info(f"   {idx}. '{text[:70]}{'...' if len(text) > 70 else ''}'")
            logger.info(f"      BERT: {bert_name}, Rule: {rule_name} (истина: {true_name})")
    
    # Какая модель лучше?
    if bert_results['accuracy'] > rule_results['accuracy']:
        winner = "BERT модель"
        advantage = bert_results['accuracy'] - rule_results['accuracy']
    elif rule_results['accuracy'] > bert_results['accuracy']:
        winner = "Rule-based модель"
        advantage = rule_results['accuracy'] - bert_results['accuracy']
    else:
        winner = "Ничья"
        advantage = 0
    
    logger.info(f"\n🏆 ИТОГОВЫЙ РЕЗУЛЬТАТ:")
    if winner != "Ничья":
        logger.info(f"   Победитель: {winner}")
        logger.info(f"   Преимущество по точности: {advantage:.3f} ({advantage*100:.1f}%)")
    else:
        logger.info(f"   Обе модели показали одинаковую точность!")
    
    # Рекомендации
    logger.info(f"\n💡 РЕКОМЕНДАЦИИ:")
    if bert_results['accuracy'] > 0.9:
        logger.info("   ✅ BERT модель показывает отличные результаты")
    if rule_results['accuracy'] > 0.9:
        logger.info("   ✅ Rule-based подход очень эффективен")
    if rule_results['prediction_time'] < bert_results['prediction_time']:
        logger.info("   ⚡ Rule-based модель работает быстрее")
    if agreement_rate > 0.9:
        logger.info("   🤝 Высокая согласованность моделей - хороший знак")
    
    logger.info("\n🎉 СРАВНЕНИЕ ЗАВЕРШЕНО!")
    logger.info("✨ Теперь у вас есть две работающие модели для детекции!")


if __name__ == "__main__":
    main() 