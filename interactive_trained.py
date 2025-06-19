#!/usr/bin/env python3
"""
Интерактивный детектор оскорбления чувств верующих (обученная модель)
"""

import sys
sys.path.append('.')
from safe_load_model import SafeReligiousDetector

def main():
    print("\n🔥 ИНТЕРАКТИВНЫЙ ДЕТЕКТОР РЕЛИГИОЗНОГО КОНТЕНТА 🔥")
    print("🎯 ОБУЧЕННАЯ МОДЕЛЬ - ModernBERT")
    print("📋 Анализирует тексты на предмет оскорбления чувств верующих")
    print("="*60)
    
    # Загрузка обученной модели
    detector = SafeReligiousDetector()
    print("📚 Загружаю обученную модель...")
    detector.load_model()
    
    print("\n📝 ИНСТРУКЦИЯ:")
    print("• Введите текст для анализа")
    print("• Введите 'exit' для завершения")
    print("• Введите 'examples' для тестовых примеров")
    print("• Введите 'stats' для статистики модели")
    print("-"*60)
    
    # Примеры для демонстрации
    examples = [
        "Религия помогает людям найти смысл жизни",
        "Все верующие - дураки и фанатики", 
        "Духовные практики могут быть полезными",
        "Церкви должны быть сожжены дотла",
        "Каждый имеет право на свою веру",
        "Религия - это яд для общества",
        "Молитва приносит внутренний покой",
        "Религиозные лидеры должны быть уничтожены"
    ]
    
    while True:
        try:
            text = input("\n🔍 Введите текст: ").strip()
            
            if text.lower() in ['exit', 'quit', 'выход']:
                print("👋 До свидания!")
                break
                
            if text.lower() in ['examples', 'примеры']:
                print("\n🧪 Тестирую на примерах:")
                for i, example in enumerate(examples, 1):
                    print(f"\n{i}. Текст: {example}")
                    pred, prob = detector.predict(example)
                    
                    safe_prob = prob[0] * 100
                    dangerous_prob = prob[1] * 100
                    confidence = max(prob) * 100
                    
                    if pred == 1:
                        print("   🚨 ОПАСНЫЙ КОНТЕНТ")
                    else:
                        print("   ✅ БЕЗОПАСНЫЙ КОНТЕНТ")
                    print(f"   Безопасный: {safe_prob:.1f}% | Опасный: {dangerous_prob:.1f}%")
                    print(f"   Уверенность: {confidence:.1f}%")
                continue
            
            if text.lower() in ['stats', 'статистика']:
                print("\n📊 СТАТИСТИКА ОБУЧЕННОЙ МОДЕЛИ:")
                print("   📁 Модель: ./religious_detector_manual")
                print("   🎯 Архитектура: ModernBERT-base")
                print("   📚 Данные: train_diff - Лист1.csv (1800 примеров)")
                print("   🔄 Обучение: 2 эпохи")
                print("   ✅ Финальная точность: 86.4% (валидация)")
                print("   💾 Статус: Модель сохранена и готова к использованию")
                continue
                
            if not text:
                print("⚠️ Введите непустой текст")
                continue
                
            # Анализ введенного текста
            pred, prob = detector.predict(text)
            
            safe_prob = prob[0] * 100
            dangerous_prob = prob[1] * 100
            confidence = max(prob) * 100
            
            print("\n" + "="*50)
            if pred == 1:
                print("🚨 ОПАСНЫЙ КОНТЕНТ")
                print("   Может оскорблять чувства верующих")
                status_emoji = "🚨"
            else:
                print("✅ БЕЗОПАСНЫЙ КОНТЕНТ")
                print("   Не содержит оскорблений")
                status_emoji = "✅"
                
            print(f"\n📊 Детальная статистика:")
            print(f"   Безопасный:  {safe_prob:5.1f}% {'█' * int(safe_prob/5)}")
            print(f"   Опасный:     {dangerous_prob:5.1f}% {'█' * int(dangerous_prob/5)}")
            print(f"\n🎯 Уверенность: {confidence:.1f}%")
            
            # Интерпретация уверенности
            if confidence >= 90:
                print("💪 Очень высокая уверенность")
                reliability = "🟢 ВЫСОКАЯ"
            elif confidence >= 75:
                print("👍 Высокая уверенность")
                reliability = "🟡 СРЕДНЯЯ" 
            elif confidence >= 60:
                print("🤔 Умеренная уверенность")
                reliability = "🟠 НИЗКАЯ"
            else:
                print("⚠️ Низкая уверенность - рекомендуется ручная проверка")
                reliability = "🔴 ОЧЕНЬ НИЗКАЯ"
            
            print(f"🔍 Надежность: {reliability}")
            print("="*50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Завершение работы...")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            print("🔄 Попробуйте еще раз")

if __name__ == "__main__":
    main() 