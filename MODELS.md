# Модели для детекции религиозного hate speech

Из-за ограничений GitHub (100MB на файл), обученные модели не включены в репозиторий. Однако вы можете легко обучить их заново или загрузить предобученные версии.

## 📁 Структура моделей

```
models/
├── bert_religious_classifier/           # Базовая BERT модель (57.5% точность)
├── optimized_religious_classifier/      # Оптимизированная модель (99.0% точность)
└── bert_religious_classifier_improved/  # Улучшенная модель (98.6% точность)
```

## 🚀 Обучение моделей

### 1. Базовая BERT модель
```bash
python scripts/train_bert.py
```

### 2. Оптимизированная модель
```bash
python train_final_model.py
```

### 3. Улучшенная модель (рекомендуется)
```bash
python train_bert_detector.py
```

## 📊 Характеристики моделей

| Модель | Архитектура | Параметры | Точность (train) | Точность (test) | Размер |
|--------|-------------|-----------|------------------|-----------------|--------|
| **BERT v1** | `cointegrated/rubert-tiny2` | 29M | 57.5% | 57.5% | ~111MB |
| **BERT v2** | `cointegrated/rubert-tiny2` | 29M | 99.0% | 65.6% | ~111MB |
| **BERT v3** | `DeepPavlov/rubert-base-cased` | 180M | 98.6% | 70.6% | ~678MB |

## 🔧 Требования для обучения

- **GPU**: NVIDIA A100 80GB (рекомендуется) или аналогичная
- **RAM**: 16GB+ системной памяти
- **Время обучения**: 
  - BERT v1/v2: ~10-15 минут
  - BERT v3: ~30 секунд (с GPU)
- **Зависимости**: см. `requirements.txt`

## 💾 Загрузка обученных моделей

После обучения модели сохраняются в соответствующих папках:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Загрузка улучшенной модели
tokenizer = AutoTokenizer.from_pretrained('./bert_religious_classifier_improved')
model = AutoModelForSequenceClassification.from_pretrained('./bert_religious_classifier_improved')
```

## 🎯 Использование

```python
# Пример использования обученной модели
import torch

def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1)
    return "Опасный" if prediction.item() == 1 else "Безопасный"

# Тестирование
result = predict("Пример текста для классификации", model, tokenizer)
print(result)
```

## 📈 Сравнение с Rule-based

Для сравнения с rule-based подходом используйте:

```bash
python compare_bert_vs_rules.py
```

**Результат**: Rule-based модель показывает 100% точность при значительно меньших ресурсных затратах.

## 🔄 Воспроизведение результатов

1. Обучите модели с помощью соответствующих скриптов
2. Запустите сравнение: `python compare_bert_vs_rules.py`
3. Результаты должны совпадать с указанными в README.md

## 📝 Примечания

- Модели обучались на датасете из 1,800 примеров
- Тестирование проводилось на 160 примерах с автоматической разметкой
- Все эксперименты воспроизводимы при использовании фиксированного seed=42 