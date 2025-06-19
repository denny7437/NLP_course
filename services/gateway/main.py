"""
Gateway Service - главный API сервис системы фильтрации
"""

import asyncio
import time
import hashlib
import uuid
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import httpx

from .models import (
    FilterRequest, FilterResponse, FilterBatchRequest, FilterBatchResponse,
    HealthResponse, LogsRequest, LogsResponse, StatsResponse, ErrorResponse
)
from .auth import get_current_user, RateLimiter
from .clients import PreFilterClient, SafetyLLMClient
from .aggregator import VerdictAggregator
from .logger import LogWriter

logger = logging.getLogger(__name__)

# Глобальные клиенты и сервисы
prefilter_client: Optional[PreFilterClient] = None
safety_llm_client: Optional[SafetyLLMClient] = None
verdict_aggregator: Optional[VerdictAggregator] = None
log_writer: Optional[LogWriter] = None
rate_limiter: Optional[RateLimiter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Startup
    logger.info("Запуск Gateway сервиса...")
    
    global prefilter_client, safety_llm_client, verdict_aggregator, log_writer, rate_limiter
    
    try:
        # Инициализация клиентов
        prefilter_client = PreFilterClient("http://prefilter:50051")
        safety_llm_client = SafetyLLMClient("http://safety-llm:8000")
        
        # Инициализация аггрегатора и логгера
        verdict_aggregator = VerdictAggregator()
        log_writer = LogWriter()
        
        # Инициализация rate limiter
        rate_limiter = RateLimiter()
        
        # Проверка подключений
        await prefilter_client.health_check()
        await safety_llm_client.health_check()
        
        logger.info("Gateway сервис успешно запущен")
        
        yield
        
    except Exception as e:
        logger.error(f"Ошибка инициализации Gateway: {e}")
        raise
    
    # Shutdown
    logger.info("Остановка Gateway сервиса...")
    
    if prefilter_client:
        await prefilter_client.close()
    if safety_llm_client:
        await safety_llm_client.close()
    if log_writer:
        await log_writer.close()
    
    logger.info("Gateway сервис остановлен")


# Создание FastAPI приложения
app = FastAPI(
    title="LLM Content Filter API",
    description="Система фильтрации LLM контента с BERT Pre-Filter + Safety-LLM",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене ограничить
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # В продакшене ограничить
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware для измерения времени обработки"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Обработчик HTTP исключений"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
            "request_id": str(uuid.uuid4())
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Обработчик общих исключений"""
    logger.error(f"Необработанная ошибка: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Внутренняя ошибка сервера",
            "error_code": "INTERNAL_ERROR",
            "request_id": str(uuid.uuid4())
        }
    )


# Основные эндпоинты

@app.post("/v1/filter", response_model=FilterResponse)
async def filter_text(
    request: FilterRequest,
    http_request: Request,
    user = Depends(get_current_user)
):
    """Фильтрация текста"""
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()
    
    try:
        # Rate limiting
        client_ip = http_request.client.host
        await rate_limiter.check_rate_limit(user.user_id, client_ip)
        
        logger.info(f"Запрос фильтрации [{request_id}]: {len(request.text)} символов")
        
        # 1. Pre-Filter
        prefilter_result = await prefilter_client.predict(request.text, request_id)
        
        if prefilter_result['safe'] and not prefilter_result['uncertain']:
            # Быстрый путь - возвращаем safe
            processing_time = (time.perf_counter() - start_time) * 1000
            
            response = FilterResponse(
                status="safe",
                processing_ms=processing_time,
                request_id=request_id
            )
            
            # Логирование
            await log_writer.log_request(
                request_id=request_id,
                text=request.text,
                status="safe",
                latency_ms=int(processing_time),
                pre_score=prefilter_result['score'],
                source_ip=client_ip,
                llm_id=request.llm_id
            )
            
            return response
        
        # 2. Safety-LLM для unsafe/uncertain случаев
        safety_result = await safety_llm_client.analyze(
            text=request.text,
            request_id=request_id,
            pre_score=prefilter_result['score']
        )
        
        # 3. Агрегация результатов
        final_verdict = verdict_aggregator.aggregate(
            prefilter_result=prefilter_result,
            safety_result=safety_result
        )
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        response = FilterResponse(
            status=final_verdict['status'],
            category=final_verdict.get('category'),
            comment=final_verdict.get('comment'),
            processing_ms=processing_time,
            request_id=request_id
        )
        
        # Логирование
        await log_writer.log_request(
            request_id=request_id,
            text=request.text,
            status=final_verdict['status'],
            category=final_verdict.get('category'),
            latency_ms=int(processing_time),
            pre_score=prefilter_result['score'],
            safety_score=safety_result.get('confidence'),
            source_ip=client_ip,
            llm_id=request.llm_id
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Ошибка обработки запроса [{request_id}]: {e}")
        processing_time = (time.perf_counter() - start_time) * 1000
        
        # Логирование ошибки
        await log_writer.log_error(request_id, str(e), processing_time)
        
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обработки запроса: {str(e)}"
        )


@app.post("/v1/filter/batch", response_model=FilterBatchResponse)
async def filter_batch(
    request: FilterBatchRequest,
    http_request: Request,
    user = Depends(get_current_user)
):
    """Батч фильтрация текстов"""
    client_ip = http_request.client.host
    
    # Rate limiting для батч запросов
    await rate_limiter.check_rate_limit(user.user_id, client_ip, multiplier=len(request.requests))
    
    # Обработка всех запросов параллельно
    tasks = []
    for req in request.requests:
        task = filter_single_request(req, client_ip, user)
        tasks.append(task)
    
    responses = await asyncio.gather(*tasks)
    
    return FilterBatchResponse(responses=responses)


async def filter_single_request(request: FilterRequest, client_ip: str, user) -> FilterResponse:
    """Обработка одного запроса в батче"""
    # Реализация аналогична filter_text, но без rate limiting
    # (он уже проверен на уровне батча)
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()
    
    try:
        # Pre-Filter
        prefilter_result = await prefilter_client.predict(request.text, request_id)
        
        if prefilter_result['safe'] and not prefilter_result['uncertain']:
            processing_time = (time.perf_counter() - start_time) * 1000
            
            response = FilterResponse(
                status="safe",
                processing_ms=processing_time,
                request_id=request_id
            )
            
            await log_writer.log_request(
                request_id=request_id,
                text=request.text,
                status="safe",
                latency_ms=int(processing_time),
                pre_score=prefilter_result['score'],
                source_ip=client_ip,
                llm_id=request.llm_id
            )
            
            return response
        
        # Safety-LLM
        safety_result = await safety_llm_client.analyze(
            text=request.text,
            request_id=request_id,
            pre_score=prefilter_result['score']
        )
        
        # Агрегация
        final_verdict = verdict_aggregator.aggregate(
            prefilter_result=prefilter_result,
            safety_result=safety_result
        )
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        response = FilterResponse(
            status=final_verdict['status'],
            category=final_verdict.get('category'),
            comment=final_verdict.get('comment'),
            processing_ms=processing_time,
            request_id=request_id
        )
        
        await log_writer.log_request(
            request_id=request_id,
            text=request.text,
            status=final_verdict['status'],
            category=final_verdict.get('category'),
            latency_ms=int(processing_time),
            pre_score=prefilter_result['score'],
            safety_score=safety_result.get('confidence'),
            source_ip=client_ip,
            llm_id=request.llm_id
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Ошибка обработки батч запроса [{request_id}]: {e}")
        processing_time = (time.perf_counter() - start_time) * 1000
        
        await log_writer.log_error(request_id, str(e), processing_time)
        
        # Возвращаем ошибку как unsafe для безопасности
        return FilterResponse(
            status="unsafe",
            category="error",
            comment=f"Ошибка обработки: {str(e)}",
            processing_ms=processing_time,
            request_id=request_id
        )


@app.get("/v1/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья системы"""
    components = {}
    overall_status = "healthy"
    
    try:
        # Проверка Pre-Filter
        prefilter_health = await prefilter_client.health_check()
        components["prefilter"] = prefilter_health
        
        # Проверка Safety-LLM
        safety_health = await safety_llm_client.health_check()
        components["safety_llm"] = safety_health
        
        # Проверка ClickHouse
        clickhouse_health = await log_writer.health_check()
        components["clickhouse"] = clickhouse_health
        
        # Определение общего статуса
        for comp_health in components.values():
            if comp_health.get("status") != "healthy":
                overall_status = "degraded"
                break
                
    except Exception as e:
        logger.error(f"Ошибка проверки здоровья: {e}")
        overall_status = "unhealthy"
        components["error"] = {"status": "error", "message": str(e)}
    
    return HealthResponse(
        status=overall_status,
        timestamp=time.time(),
        version="1.0.0",
        components=components
    )


@app.get("/v1/logs", response_model=LogsResponse)
async def get_logs(
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    status: Optional[str] = None,
    category: Optional[str] = None,
    llm_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    user = Depends(get_current_user)
):
    """Получение логов фильтрации"""
    try:
        logs_request = LogsRequest(
            start_time=start_time,
            end_time=end_time,
            status=status,
            category=category,
            llm_id=llm_id,
            limit=limit,
            offset=offset
        )
        
        logs_data = await log_writer.get_logs(logs_request)
        return logs_data
        
    except Exception as e:
        logger.error(f"Ошибка получения логов: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/stats", response_model=StatsResponse)
async def get_stats(
    hours: int = 24,
    user = Depends(get_current_user)
):
    """Получение статистики системы"""
    try:
        stats = await log_writer.get_stats(hours)
        return stats
        
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 