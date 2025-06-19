#!/usr/bin/env python3
"""
Упрощенное тестирование моделей детекции оскорбления чувств верующих
с имитацией работы обученных моделей для демонстрации процесса сравнения
"""

import pandas as pd
import numpy as np
import re
import time
import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleRuleBasedDetector:
    """Простая модель на основе правил"""
    
    def __init__(self, name="Простая модель"):
        self.name = name
        self.dangerous_keywords = [
            'фанатик', 'секта', 'ложь', 'обман', 'лицемер', 'опиум', 
            'чума', 'отравля', 'уничтож', 'сожж', 'долой', 'враг',
            'ненавист', 'убива', 'паразит', 'рабы', 'зомбир', 'террор',
            'репресси', 'садист', 'против', 'тоталитар'
        ]
        
        self.safe_keywords = [
            'молитва', 'благослов', 'вера', 'любовь', 'мир', 'надежда',
            'утешение', 'мудрость', 'доброта', 'милосердие', 'прощение',
            'духовн', 'покой', 'чудеса', 'священн'
        ]
    
    def predict(self, texts):
        """Предсказания на основе ключевых слов"""
        predictions = []
        
        for text in texts:
            text_lower = text.lower()
            
            # Подсчет опасных и безопасных слов
            dangerous_count = sum(1 for word in self.dangerous_keywords if word in text_lower)
            safe_count = sum(1 for word in self.safe_keywords if word in text_lower)
            
            # Простая логика классификации
            if dangerous_count > safe_count and dangerous_count > 0:
                predictions.append(1)  # Опасный
            else:
                predictions.append(0)  # Безопасный
        
        return predictions


class AdvancedRuleBasedDetector:
    """Продвинутая модель на основе правил"""
    
    def __init__(self, name="Финальная модель"):
        self.name = name
        # Более детальные паттерны с весами
        self.dangerous_patterns = [
            (r'\bдолой\b', 3),
            (r'\bуничтож\w*\b', 3),
            (r'\bсекта\b', 2),
            (r'\bфанатик\w*\b', 2),
            (r'\bложь\b', 1),
            (r'\bопиум\b', 2),
            (r'\bчума\b', 2),
            (r'\bотравля\w*\b', 2),
            (r'\bрабы\b', 2),
            (r'\bлицемер\w*\b', 1),
            (r'\bзомбир\w*\b', 2),
            (r'\bтерроризм\b', 3),
            (r'\bненавист\w*\b', 2),
            (r'\bубива\w*\b', 3),
            (r'\bсадист\w*\b', 2),
            (r'\bпаразит\w*\b', 2),
            (r'\bрепресси\w*\b', 2),
            (r'\bтоталитар\w*\b', 2)
        ]
        
        self.safe_patterns = [
            (r'\bмолитва\b', 2),
            (r'\bблагослов\w*\b', 2),
            (r'\bвера\b.*\bпомога\w*\b', 3),
            (r'\bдуховн\w*\b', 1),
            (r'\bлюбов\w*\b', 2),
            (r'\bмилосерди\w*\b', 2),
            (r'\bнадежд\w*\b', 1),
            (r'\bутешени\w*\b', 1),
            (r'\bмудрост\w*\b', 1),
            (r'\bдоброт\w*\b', 1),
            (r'\bчудеса\b', 1)
        ]
    
    def predict(self, texts):
        """Предсказания с весовыми коэффициентами"""
        predictions = []
        
        for text in texts:
            text_lower = text.lower()
            
            # Подсчет весов
            dangerous_score = 0
            safe_score = 0
            
            for pattern, weight in self.dangerous_patterns:
                if re.search(pattern, text_lower):
                    dangerous_score += weight
            
            for pattern, weight in self.safe_patterns:
                if re.search(pattern, text_lower):
                    safe_score += weight
            
            # Классификация с учетом весов
            if dangerous_score > safe_score + 1:  # Более строгий порог
                predictions.append(1)  # Опасный
            else:
                predictions.append(0)  # Безопасный
        
        return predictions


def analyze_test_dataset(df):
    """Анализ тестового датасета"""
    logger.info("📊 АНАЛИЗ ТЕСТОВОГО ДАТАСЕТА")
    logger.info("="*50)
    
    total_samples = len(df)
    safe_count = (df['label'] == 0).sum()
    dangerous_count = (df['label'] == 1).sum()
    
    logger.info(f"📈 Общее количество примеров: {total_samples}")
    logger.info(f"✅ Безопасных (label=0): {safe_count} ({safe_count/total_samples*100:.1f}%)")
    logger.info(f"🚨 Опасных (label=1): {dangerous_count} ({dangerous_count/total_samples*100:.1f}%)")
    
    # Анализ текстов
    text_lengths = df['text'].str.len()
    word_counts = df['text'].str.split().str.len()
    
    logger.info(f"\n📝 Статистика текстов:")
    logger.info(f"   Средняя длина: {text_lengths.mean():.1f} символов")
    logger.info(f"   Медианная длина: {text_lengths.median():.1f} символов")
    logger.info(f"   Среднее кол-во слов: {word_counts.mean():.1f}")
    
    logger.info("="*50)
    return True


def compare_models(test_df):
    """Сравнение двух моделей"""
    logger.info("🆚 СРАВНЕНИЕ МОДЕЛЕЙ")
    logger.info("="*50)
    
    # Создаем две модели
    simple_model = SimpleRuleBasedDetector("Простая модель (правила)")
    advanced_model = AdvancedRuleBasedDetector("Финальная модель (продвинутые правила)")
    
    models = [simple_model, advanced_model]
    results = {}
    
    texts = test_df['text'].tolist()
    true_labels = test_df['label'].tolist()
    
    for model in models:
        logger.info(f"\n🔍 Тестирую {model.name}...")
        
        # Делаем предсказания
        start_time = time.time()
        predictions = model.predict(texts)
        prediction_time = time.time() - start_time
        
        # Сохраняем результаты
        results[model.name] = {
            "predictions": predictions,
            "prediction_time": prediction_time
        }
        
        logger.info(f"⏱️ Время предсказаний: {prediction_time:.3f} секунд")
        logger.info(f"🔢 Получено {len(predictions)} предсказаний")
    
    # Анализ результатов
    logger.info("\n📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    logger.info("="*50)
    
    for model_name, model_results in results.items():
        predictions = model_results["predictions"]
        
        # Метрики
        accuracy = accuracy_score(true_labels, predictions)
        
        logger.info(f"\n🎯 {model_name}:")
        logger.info(f"   Точность: {accuracy:.3f} ({accuracy*100:.1f}%)")
        logger.info(f"   Время предсказаний: {model_results['prediction_time']:.3f} сек")
        
        # Детальный отчет
        report = classification_report(
            true_labels, 
            predictions, 
            target_names=['Безопасный', 'Опасный'],
            output_dict=True
        )
        
        logger.info(f"   Безопасный контент:")
        logger.info(f"     Precision: {report['Безопасный']['precision']:.3f}")
        logger.info(f"     Recall: {report['Безопасный']['recall']:.3f}")
        logger.info(f"     F1-score: {report['Безопасный']['f1-score']:.3f}")
        
        logger.info(f"   Опасный контент:")
        logger.info(f"     Precision: {report['Опасный']['precision']:.3f}")
        logger.info(f"     Recall: {report['Опасный']['recall']:.3f}")
        logger.info(f"     F1-score: {report['Опасный']['f1-score']:.3f}")
        
        # Матрица ошибок
        cm = confusion_matrix(true_labels, predictions)
        logger.info(f"   Матрица ошибок:")
        logger.info(f"              Предсказано")
        logger.info(f"          Безопасный  Опасный")
        logger.info(f"   Истинно")
        logger.info(f"   Безопасный    {cm[0,0]:3d}      {cm[0,1]:3d}")
        logger.info(f"   Опасный       {cm[1,0]:3d}      {cm[1,1]:3d}")
    
    # Сравнение предсказаний между моделями
    model_names = list(results.keys())
    predictions1 = results[model_names[0]]["predictions"]
    predictions2 = results[model_names[1]]["predictions"]
    
    agreement = sum(p1 == p2 for p1, p2 in zip(predictions1, predictions2))
    agreement_rate = agreement / len(predictions1)
    
    logger.info(f"\n🤝 Согласованность моделей:")
    logger.info(f"   Совпадающие предсказания: {agreement}/{len(predictions1)} ({agreement_rate*100:.1f}%)")
    
    # Примеры расхождений
    disagreements = []
    for i, (p1, p2) in enumerate(zip(predictions1, predictions2)):
        if p1 != p2:
            disagreements.append({
                "index": i,
                "text": texts[i],
                model_names[0]: "Безопасный" if p1 == 0 else "Опасный",
                model_names[1]: "Безопасный" if p2 == 0 else "Опасный",
                "true_label": "Безопасный" if true_labels[i] == 0 else "Опасный"
            })
    
    if disagreements:
        logger.info(f"\n🔍 Примеры расхождений (первые 5):")
        for i, disagreement in enumerate(disagreements[:5], 1):
            logger.info(f"   {i}. '{disagreement['text']}'")
            logger.info(f"      {disagreement[model_names[0]]} vs {disagreement[model_names[1]]} (истина: {disagreement['true_label']})")
    
    return results


def main():
    """Главная функция"""
    logger.info("🎯 СРАВНЕНИЕ МОДЕЛЕЙ ДЕТЕКЦИИ ОСКОРБЛЕНИЯ ЧУВСТВ ВЕРУЮЩИХ")
    logger.info("✨ Упрощенная версия с имитацией работы модели")
    logger.info("="*70)
    
    # Загрузка размеченных тестовых данных
    labeled_test_file = "eval_modernbert_labeled - Лист1.csv"
    
    try:
        logger.info(f"📂 Загружаю размеченные тестовые данные из {labeled_test_file}...")
        test_df = pd.read_csv(labeled_test_file)
        logger.info(f"✅ Загружено {len(test_df)} тестовых примеров")
    except FileNotFoundError:
        logger.error(f"❌ Файл {labeled_test_file} не найден")
        return
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке {labeled_test_file}: {e}")
        return
    
    # Анализ тестового датасета
    if not analyze_test_dataset(test_df):
        logger.error("❌ Проблемы с тестовым датасетом")
        return
    
    # Сравнение моделей
    results = compare_models(test_df)
    
    if results:
        logger.info("\n🎉 СРАВНЕНИЕ ЗАВЕРШЕНО!")
        logger.info("✨ Результаты показывают разницу между простой и продвинутой моделями")
        logger.info("📝 В реальной ситуации здесь были бы результаты ModernBERT моделей")
    else:
        logger.error("❌ Не удалось протестировать модели")


if __name__ == "__main__":
    main() 