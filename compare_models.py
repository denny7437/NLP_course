#!/usr/bin/env python3
"""
Сравнение производительности двух моделей детекции оскорбления чувств верующих
на тестовой выборке eval_modernbert - Лист1.csv
"""

import pandas as pd
import numpy as np
import subprocess
import os
import sys
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTester:
    """Класс для тестирования моделей"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.max_length = 512
    
    def load_model(self):
        """Загрузка модели"""
        try:
            logger.info(f"🔄 Загружаю модель из {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()
            logger.info("✅ Модель загружена успешно")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            return False
    
    def predict(self, texts):
        """Предсказания для списка текстов"""
        predictions = []
        
        logger.info(f"🔮 Делаю предсказания для {len(texts)} текстов...")
        
        # Принудительно используем CPU
        device = torch.device('cpu')
        self.model.to(device)
        
        with torch.no_grad():
            for i, text in enumerate(texts):
                if i % 100 == 0:
                    logger.info(f"   Обработано: {i}/{len(texts)}")
                
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length
                )
                
                # Переносим входные данные на CPU
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                predicted_class = torch.argmax(outputs.logits, dim=-1)
                predictions.append(predicted_class.item())
        
        logger.info("✅ Предсказания завершены")
        return predictions


def run_training_script(script_name):
    """Запуск скрипта обучения"""
    logger.info(f"🚀 Запускаю обучение: {script_name}")
    
    try:
        result = subprocess.run([
            sys.executable, script_name
        ], capture_output=True, text=True, timeout=3600)  # 1 час timeout
        
        if result.returncode == 0:
            logger.info(f"✅ Обучение {script_name} завершено успешно")
            logger.info("Последние строки вывода:")
            for line in result.stdout.split('\n')[-10:]:
                if line.strip():
                    logger.info(f"   {line}")
            return True
        else:
            logger.error(f"❌ Ошибка в {script_name}:")
            logger.error(f"   Stdout: {result.stdout}")
            logger.error(f"   Stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Превышен timeout для {script_name}")
        return False
    except Exception as e:
        logger.error(f"💥 Исключение при запуске {script_name}: {e}")
        return False


def analyze_test_dataset(df):
    """Анализ тестового датасета"""
    logger.info("📊 АНАЛИЗ ТЕСТОВОГО ДАТАСЕТА")
    logger.info("="*50)
    
    # Проверяем наличие колонки label
    if 'label' not in df.columns:
        logger.warning("⚠️ В тестовом датасете нет колонки 'label'")
        logger.info("🔍 Доступные колонки:", list(df.columns))
        return False
    
    total_samples = len(df)
    safe_count = (df['label'] == 0).sum() if 'label' in df.columns else 0
    dangerous_count = (df['label'] == 1).sum() if 'label' in df.columns else 0
    
    logger.info(f"📈 Общее количество примеров: {total_samples}")
    if 'label' in df.columns:
        logger.info(f"✅ Безопасных (label=0): {safe_count} ({safe_count/total_samples*100:.1f}%)")
        logger.info(f"🚨 Опасных (label=1): {dangerous_count} ({dangerous_count/total_samples*100:.1f}%)")
    
    # Анализ текстов
    text_lengths = df['text'].str.len()
    word_counts = df['text'].str.split().str.len()
    
    logger.info(f"\n📝 Статистика текстов:")
    logger.info(f"   Средняя длина: {text_lengths.mean():.1f} символов")
    logger.info(f"   Медианная длина: {text_lengths.median():.1f} символов")
    logger.info(f"   Среднее кол-во слов: {word_counts.mean():.1f}")
    
    # Примеры
    logger.info(f"\n🔍 Первые 3 примера:")
    for i, text in enumerate(df['text'].head(3), 1):
        label_info = f" (label={df.iloc[i-1]['label']})" if 'label' in df.columns else ""
        logger.info(f"   {i}.{label_info} {text}")
    
    logger.info("="*50)
    return True


def compare_models(test_df):
    """Сравнение двух моделей"""
    logger.info("🆚 СРАВНЕНИЕ МОДЕЛЕЙ")
    logger.info("="*50)
    
    # Пути к моделям
    simple_model_path = "./temp_religious_classifier"  # из train_religious_detector_simple.py
    final_model_path = "./religious_content_detector_final"  # из train_final_model.py
    
    models_info = {
        "Простая модель": {
            "path": simple_model_path,
            "script": "train_religious_detector_simple.py"
        },
        "Финальная модель": {
            "path": final_model_path,
            "script": "train_final_model.py"
        }
    }
    
    results = {}
    texts = test_df['text'].tolist()
    true_labels = test_df['label'].tolist() if 'label' in test_df.columns else None
    
    for model_name, model_info in models_info.items():
        logger.info(f"\n🔍 Тестирую {model_name}...")
        
        # Проверяем существование модели
        if not os.path.exists(model_info["path"]):
            logger.warning(f"⚠️ Модель {model_name} не найдена по пути {model_info['path']}")
            logger.info(f"🔄 Запускаю обучение...")
            
            if not run_training_script(model_info["script"]):
                logger.error(f"❌ Не удалось обучить {model_name}")
                continue
        
        # Загружаем и тестируем модель
        tester = ModelTester(model_info["path"])
        if not tester.load_model():
            logger.error(f"❌ Не удалось загрузить {model_name}")
            continue
        
        # Делаем предсказания
        start_time = time.time()
        predictions = tester.predict(texts)
        prediction_time = time.time() - start_time
        
        # Сохраняем результаты
        results[model_name] = {
            "predictions": predictions,
            "prediction_time": prediction_time,
            "model_path": model_info["path"]
        }
        
        logger.info(f"⏱️ Время предсказаний: {prediction_time:.2f} секунд")
        logger.info(f"🔢 Получено {len(predictions)} предсказаний")
    
    # Анализ результатов
    if true_labels and len(results) >= 2:
        logger.info("\n📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
        logger.info("="*50)
        
        for model_name, model_results in results.items():
            predictions = model_results["predictions"]
            
            # Метрики
            accuracy = accuracy_score(true_labels, predictions)
            
            logger.info(f"\n🎯 {model_name}:")
            logger.info(f"   Точность: {accuracy:.3f} ({accuracy*100:.1f}%)")
            logger.info(f"   Время предсказаний: {model_results['prediction_time']:.2f} сек")
            
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
    if len(results) >= 2:
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
                    "true_label": "Безопасный" if true_labels[i] == 0 else "Опасный" if true_labels else "Неизвестно"
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
    logger.info("="*70)
    
    # Загрузка тестовых данных
    test_file = "eval_modernbert - Лист1.csv"
    labeled_test_file = "eval_modernbert_labeled - Лист1.csv"
    
    # Проверяем, есть ли уже размеченные данные
    if not os.path.exists(labeled_test_file):
        logger.info("📝 Размеченный тестовый файл не найден, создаю разметку...")
        try:
            result = subprocess.run([sys.executable, "label_test_data.py"], 
                                  capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                logger.error(f"❌ Ошибка при создании разметки: {result.stderr}")
                return
            logger.info("✅ Разметка создана успешно")
        except Exception as e:
            logger.error(f"❌ Ошибка при запуске разметки: {e}")
            return
    
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
        logger.info("✨ Результаты сохранены в логах выше")
    else:
        logger.error("❌ Не удалось протестировать модели")


if __name__ == "__main__":
    main() 