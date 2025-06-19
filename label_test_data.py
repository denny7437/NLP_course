#!/usr/bin/env python3
"""
Автоматическая разметка тестовых данных eval_modernbert - Лист1.csv
"""

import pandas as pd
import re

def automatic_labeling(text):
    """
    Автоматическая разметка текста на основе ключевых слов и паттернов
    0 - безопасный контент
    1 - потенциально опасный контент
    """
    text_lower = text.lower()
    
    # Очень явные негативные паттерны - опасный контент
    dangerous_patterns = [
        r'\bдолой\b',
        r'\bуничтож\w*\b',
        r'\bзапрет\w*\b',
        r'\bсожж\w*\b',
        r'\bраковая опухоль\b',
        r'\bчума\b',
        r'\bотравля\w*\b',
        r'\bрабы\b',
        r'\bфанатик\w*\b',
        r'\bсекта\b',
        r'\bсектант\w*\b', 
        r'\bпсихическое расстройство\b',
        r'\bопиум\b',
        r'\bлицемер\w*\b',
        r'\bобман\w*\b',
        r'\bложь\b',
        r'\bмозги промыва\w*\b',
        r'\bзомбир\w*\b',
        r'\bтерроризм\b',
        r'\bкровь\b.*\bрелиги\w*\b',
        r'\bненавист\w*\b',
        r'\bвраг\w*\b',
        r'\bубива\w*\b',
        r'\bсадист\w*\b',
        r'\bпаразит\w*\b',
        r'\bрепресси\w*\b',
        r'\bтоталитар\w*\b'
    ]
    
    # Очень явные позитивные паттерны - безопасный контент
    safe_patterns = [
        r'\bмолитва\b',
        r'\bблагослов\w*\b',
        r'\bвера\b.*\bпомога\w*\b',
        r'\bдуховн\w*\b.*\bразвити\w*\b',
        r'\bмир\w*\b.*\bсогласи\w*\b',
        r'\bлюбов\w*\b',
        r'\bмилосерди\w*\b',
        r'\bпрощени\w*\b',
        r'\bнадежд\w*\b',
        r'\bутешени\w*\b',
        r'\bмудрост\w*\b',
        r'\bдоброт\w*\b'
    ]
    
    # Проверяем на опасные паттерны
    for pattern in dangerous_patterns:
        if re.search(pattern, text_lower):
            return 1
    
    # Проверяем на безопасные паттерны
    for pattern in safe_patterns:
        if re.search(pattern, text_lower):
            return 0
    
    # Дополнительная логика по контексту
    if any(word in text_lower for word in ['христиан', 'религи', 'церков', 'вер', 'бог']):
        # Контент связан с религией
        negative_words = ['против', 'плохо', 'вред', 'зло', 'неправильно', 'критик']
        if any(word in text_lower for word in negative_words):
            return 1
        else:
            return 0
    
    # По умолчанию - безопасный
    return 0

def main():
    # Загружаем данные
    df = pd.read_csv("eval_modernbert - Лист1.csv")
    print(f"Загружено {len(df)} примеров для разметки")
    
    # Добавляем автоматические метки
    df['label'] = df['text'].apply(automatic_labeling)
    
    # Статистика
    safe_count = (df['label'] == 0).sum()
    dangerous_count = (df['label'] == 1).sum()
    
    print(f"Разметка завершена:")
    print(f"  Безопасных: {safe_count} ({safe_count/len(df)*100:.1f}%)")
    print(f"  Опасных: {dangerous_count} ({dangerous_count/len(df)*100:.1f}%)")
    
    # Сохраняем размеченные данные
    output_file = "eval_modernbert_labeled - Лист1.csv"
    df.to_csv(output_file, index=False)
    print(f"Размеченные данные сохранены в {output_file}")
    
    # Показываем примеры
    print("\nПримеры разметки:")
    print("Безопасный контент:")
    for i, row in df[df['label'] == 0].head(3).iterrows():
        print(f"  - {row['text']}")
    
    print("\nОпасный контент:")
    for i, row in df[df['label'] == 1].head(3).iterrows():
        print(f"  - {row['text']}")

if __name__ == "__main__":
    main() 