"""
Pydantic модели для Gateway API
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class FilterRequest(BaseModel):
    """Запрос фильтрации текста"""
    text: str = Field(..., min_length=1, max_length=16384, description="Текст для фильтрации")
    llm_id: Optional[str] = Field(None, description="Идентификатор LLM модели")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Дополнительные метаданные")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Текст не может быть пустым')
        return v.strip()


class FilterResponse(BaseModel):
    """Ответ фильтрации"""
    status: str = Field(..., description="Статус: safe или unsafe")
    category: Optional[str] = Field(None, description="Категория опасности")
    comment: Optional[str] = Field(None, description="Комментарий или пояснение")
    processing_ms: float = Field(..., description="Время обработки в миллисекундах")
    request_id: Optional[str] = Field(None, description="Идентификатор запроса")
    
    @validator('status')
    def status_must_be_valid(cls, v):
        if v not in ['safe', 'unsafe']:
            raise ValueError('Статус должен быть safe или unsafe')
        return v


class FilterBatchRequest(BaseModel):
    """Батч запрос фильтрации"""
    requests: list[FilterRequest] = Field(..., min_items=1, max_items=100, description="Список запросов")


class FilterBatchResponse(BaseModel):
    """Батч ответ фильтрации"""
    responses: list[FilterResponse] = Field(..., description="Список ответов")


class HealthResponse(BaseModel):
    """Ответ проверки здоровья"""
    status: str = Field(..., description="Статус сервиса")
    timestamp: float = Field(..., description="Временная метка")
    version: str = Field(..., description="Версия сервиса")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Статус компонентов")


class LogsRequest(BaseModel):
    """Запрос логов"""
    start_time: Optional[float] = Field(None, description="Начальное время (Unix timestamp)")
    end_time: Optional[float] = Field(None, description="Конечное время (Unix timestamp)")
    status: Optional[str] = Field(None, description="Фильтр по статусу")
    category: Optional[str] = Field(None, description="Фильтр по категории")
    llm_id: Optional[str] = Field(None, description="Фильтр по LLM ID")
    limit: int = Field(100, ge=1, le=1000, description="Лимит записей")
    offset: int = Field(0, ge=0, description="Смещение")


class LogEntry(BaseModel):
    """Запись лога"""
    timestamp: float = Field(..., description="Временная метка")
    request_id: str = Field(..., description="ID запроса")
    text_hash: str = Field(..., description="Хеш текста")
    status: str = Field(..., description="Статус фильтрации")
    category: Optional[str] = Field(None, description="Категория")
    llm_id: Optional[str] = Field(None, description="LLM ID")
    latency_ms: int = Field(..., description="Задержка в мс")
    pre_score: Optional[float] = Field(None, description="Скор Pre-Filter")
    safety_score: Optional[float] = Field(None, description="Скор Safety-LLM")
    source_ip: str = Field(..., description="IP источника")
    token_count: Optional[int] = Field(None, description="Количество токенов")


class LogsResponse(BaseModel):
    """Ответ с логами"""
    logs: list[LogEntry] = Field(..., description="Список записей логов")
    total: int = Field(..., description="Общее количество записей")
    has_more: bool = Field(..., description="Есть ли еще записи")


class StatsResponse(BaseModel):
    """Статистика системы"""
    total_requests: int = Field(..., description="Всего запросов")
    safe_requests: int = Field(..., description="Безопасных запросов") 
    unsafe_requests: int = Field(..., description="Небезопасных запросов")
    avg_latency_ms: float = Field(..., description="Средняя задержка")
    p95_latency_ms: float = Field(..., description="P95 задержка")
    rps: float = Field(..., description="Запросов в секунду")
    error_rate: float = Field(..., description="Процент ошибок")
    prefilter_hit_rate: float = Field(..., description="Процент попаданий в Pre-Filter")


class ErrorResponse(BaseModel):
    """Ответ с ошибкой"""
    error: str = Field(..., description="Сообщение об ошибке")
    error_code: Optional[str] = Field(None, description="Код ошибки")
    details: Optional[Dict[str, Any]] = Field(None, description="Дополнительные детали")
    request_id: Optional[str] = Field(None, description="ID запроса") 