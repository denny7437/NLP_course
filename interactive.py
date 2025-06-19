#!/usr/bin/env python3
"""
Интерактивный детектор оскорбления чувств верующих
"""

import sys
import os
sys.path.append('.')
from test_religious_detector import ReligiousContentTester
import logging

# Убираем лишние логи
logging.basicConfig(level=logging.WARNING)

def main():
    print("\n🔥 ИНТЕРАКТИВНЫЙ ДЕТЕКТОР РЕЛИГИОЗНОГО КОНТЕНТА 🔥")
    print("📋 Анализирует тексты на предмет оскорбления чувств верующих")
    print("🤖 Powered by ModernBERT")
    print("="*60)
    
    detector = ReligiousContentTester()
    print("📚 Загружаю модель (это может занять время)...")
    detector.initialize_model()
    print("✅ Модель готова!")
    
    print("\n📝 ИНСТРУКЦИЯ:")
    print("• Введите текст для анализа")
    print("• Введите 'exit' для завершения")
    print("• Введите 'examples' для тестовых примеров")
    print("-"*60)
    
    examples = [
        "Молитва помогает найти внутренний покой",
        "Все верующие - дураки и фанатики", 
        "Различные религии имеют право на существование",
        "Церковь должна быть сожжена дотла",
        "Духовные практики полезны для здоровья",
        "Религиозные книги - сплошная ложь"
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
                    predictions, probabilities = detector.predict([example])
                    pred = predictions[0]
                    prob = probabilities[0]
                    
                    safe_prob = prob[0] * 100
                    dangerous_prob = prob[1] * 100
                    
                    if pred == 1:
                        print("   🚨 ОПАСНЫЙ КОНТЕНТ")
                    else:
                        print("   ✅ БЕЗОПАСНЫЙ КОНТЕНТ")
                    print(f"   Безопасный: {safe_prob:.1f}% | Опасный: {dangerous_prob:.1f}%")
                continue
                
            if not text:
                print("⚠️ Введите непустой текст")
                continue
                
            # Анализ введенного текста
            predictions, probabilities = detector.predict([text])
            pred = predictions[0]
            prob = probabilities[0]
            
            safe_prob = prob[0] * 100
            dangerous_prob = prob[1] * 100
            confidence = max(prob) * 100
            
            print("\n" + "="*50)
            if pred == 1:
                print("🚨 ОПАСНЫЙ КОНТЕНТ")
                print("   Может оскорблять чувства верующих")
            else:
                print("✅ БЕЗОПАСНЫЙ КОНТЕНТ")
                print("   Не содержит оскорблений")
                
            print(f"\n📊 Детальная статистика:")
            print(f"   Безопасный:  {safe_prob:5.1f}% {'█' * int(safe_prob/5)}")
            print(f"   Опасный:     {dangerous_prob:5.1f}% {'█' * int(dangerous_prob/5)}")
            print(f"\n🎯 Уверенность: {confidence:.1f}%")
            
            if confidence >= 80:
                print("💪 Высокая уверенность в результате")
            elif confidence >= 60:
                print("🤔 Средняя уверенность - рекомендуется проверка")
            else:
                print("⚠️ Низкая уверенность - требуется ручная проверка")
            
            print("="*50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Завершение работы...")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            print("🔄 Попробуйте еще раз")

if __name__ == "__main__":
    main() 