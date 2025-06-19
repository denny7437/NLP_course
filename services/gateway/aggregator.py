"""
Агрегатор решений для объединения результатов фильтрации
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class VerdictAggregator:
    """Агрегатор для объединения результатов Pre-Filter и Safety-LLM"""
    
    def __init__(self):
        # Пороги для принятия решений
        self.confidence_threshold = 0.7
        self.prefilter_weight = 0.3
        self.safety_llm_weight = 0.7
        
    def aggregate(self, prefilter_result: Dict[str, Any], 
                  safety_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Агрегация результатов Pre-Filter и Safety-LLM
        
        Логика:
        1. Если Safety-LLM определенно говорит unsafe - возвращаем unsafe
        2. Если Safety-LLM говорит safe с высокой уверенностью - возвращаем safe
        3. При неопределенности учитываем оба результата
        """
        try:
            # Извлекаем основные поля
            prefilter_safe = prefilter_result.get('safe', False)
            prefilter_score = prefilter_result.get('score', 0.0)
            
            safety_status = safety_result.get('status', 'unsafe')
            safety_confidence = safety_result.get('confidence', 0.0)
            safety_category = safety_result.get('category')
            safety_reasoning = safety_result.get('reasoning', '')
            
            # Основное решение принимает Safety-LLM как более точная модель
            if safety_status == 'unsafe':
                final_status = 'unsafe'
                comment = self._build_comment(
                    decision="unsafe_by_safety_llm",
                    safety_reasoning=safety_reasoning,
                    safety_confidence=safety_confidence,
                    prefilter_score=prefilter_score
                )
                category = safety_category or "harmful_content"
                
            elif safety_status == 'safe' and safety_confidence >= self.confidence_threshold:
                final_status = 'safe'
                comment = self._build_comment(
                    decision="safe_by_safety_llm",
                    safety_confidence=safety_confidence,
                    prefilter_score=prefilter_score
                )
                category = None
                
            else:
                # Случай неопределенности - взвешенное решение
                combined_score = self._calculate_combined_score(
                    prefilter_score, safety_confidence, safety_status
                )
                
                if combined_score >= self.confidence_threshold:
                    final_status = 'safe'
                    comment = self._build_comment(
                        decision="safe_by_combined",
                        combined_score=combined_score,
                        prefilter_score=prefilter_score,
                        safety_confidence=safety_confidence
                    )
                    category = None
                else:
                    final_status = 'unsafe'
                    comment = self._build_comment(
                        decision="unsafe_by_combined",
                        combined_score=combined_score,
                        safety_reasoning=safety_reasoning
                    )
                    category = safety_category or "uncertain"
            
            return {
                'status': final_status,
                'category': category,
                'comment': comment,
                'confidence': safety_confidence,
                'prefilter_score': prefilter_score,
                'decision_method': self._get_decision_method(
                    safety_status, safety_confidence
                )
            }
            
        except Exception as e:
            logger.error(f"Error in verdict aggregation: {e}")
            # В случае ошибки возвращаем консервативное решение
            return {
                'status': 'unsafe',
                'category': 'aggregation_error',
                'comment': f'Ошибка агрегации результатов: {str(e)}',
                'confidence': 0.0,
                'prefilter_score': prefilter_result.get('score', 0.0)
            }
    
    def _calculate_combined_score(self, prefilter_score: float, 
                                 safety_confidence: float, 
                                 safety_status: str) -> float:
        """Расчет комбинированного скора"""
        # Преобразуем safety статус в численный скор
        if safety_status == 'safe':
            safety_score = safety_confidence
        else:
            safety_score = 1.0 - safety_confidence
        
        # Взвешенная комбинация
        combined = (
            self.prefilter_weight * prefilter_score +
            self.safety_llm_weight * safety_score
        )
        
        return max(0.0, min(1.0, combined))
    
    def _build_comment(self, decision: str, **kwargs) -> str:
        """Построение комментария к решению"""
        comments = {
            "unsafe_by_safety_llm": (
                f"Контент признан небезопасным моделью Safety-LLM "
                f"(уверенность: {kwargs.get('safety_confidence', 0):.2f}). "
                f"Причина: {kwargs.get('safety_reasoning', 'не указана')}"
            ),
            "safe_by_safety_llm": (
                f"Контент признан безопасным моделью Safety-LLM "
                f"(уверенность: {kwargs.get('safety_confidence', 0):.2f})"
            ),
            "safe_by_combined": (
                f"Контент признан безопасным на основе комбинированного анализа "
                f"(общий скор: {kwargs.get('combined_score', 0):.2f})"
            ),
            "unsafe_by_combined": (
                f"Контент признан небезопасным на основе комбинированного анализа "
                f"(общий скор: {kwargs.get('combined_score', 0):.2f}). "
                f"{kwargs.get('safety_reasoning', '')}"
            )
        }
        
        return comments.get(decision, f"Решение принято методом {decision}")
    
    def _get_decision_method(self, safety_status: str, safety_confidence: float) -> str:
        """Определение метода принятия решения"""
        if safety_status in ['safe', 'unsafe'] and safety_confidence >= self.confidence_threshold:
            return 'safety_llm_primary'
        else:
            return 'combined_scoring'
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Получение статистики агрегации"""
        return {
            'confidence_threshold': self.confidence_threshold,
            'prefilter_weight': self.prefilter_weight,
            'safety_llm_weight': self.safety_llm_weight,
            'supported_methods': [
                'safety_llm_primary',
                'combined_scoring'
            ]
        }
    
    def update_weights(self, prefilter_weight: float, safety_llm_weight: float):
        """Обновление весов для агрегации"""
        if prefilter_weight + safety_llm_weight != 1.0:
            raise ValueError("Сумма весов должна равняться 1.0")
        
        self.prefilter_weight = prefilter_weight
        self.safety_llm_weight = safety_llm_weight
        
        logger.info(f"Updated aggregation weights: pre-filter={prefilter_weight}, safety-llm={safety_llm_weight}")
    
    def update_confidence_threshold(self, threshold: float):
        """Обновление порога уверенности"""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Порог уверенности должен быть между 0.0 и 1.0")
        
        self.confidence_threshold = threshold
        logger.info(f"Updated confidence threshold: {threshold}") 