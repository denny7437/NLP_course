"""
Модуль логирования в ClickHouse
"""

import asyncio
import hashlib
import logging
import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class MockLogWriter:
    """Мок-версия LogWriter для разработки без ClickHouse"""
    
    def __init__(self):
        self.logs = []  # В реальности будет ClickHouse
        self.max_logs = 10000  # Лимит для in-memory хранения
        
    async def log_request(self, request_id: str, text: str, status: str,
                         latency_ms: int, pre_score: float, source_ip: str,
                         llm_id: Optional[str] = None, category: Optional[str] = None,
                         safety_score: Optional[float] = None):
        """Логирование запроса фильтрации"""
        try:
            # Создаем хеш текста для приватности
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            
            # Примерный подсчет токенов
            token_count = len(text.split())
            
            log_entry = {
                'ts': time.time(),
                'request_id': request_id,
                'text_hash': text_hash,
                'status': status,
                'category': category,
                'llm_id': llm_id,
                'latency_ms': latency_ms,
                'pre_score': pre_score,
                'safety_score': safety_score,
                'source_ip': source_ip,
                'token_count': token_count
            }
            
            # Добавляем в in-memory хранилище
            self.logs.append(log_entry)
            
            # Ограничиваем размер
            if len(self.logs) > self.max_logs:
                self.logs = self.logs[-self.max_logs:]
            
            logger.debug(f"Logged request {request_id}: {status}")
            
        except Exception as e:
            logger.error(f"Failed to log request {request_id}: {e}")
    
    async def log_error(self, request_id: str, error: str, latency_ms: float):
        """Логирование ошибки"""
        try:
            error_entry = {
                'ts': time.time(),
                'request_id': request_id,
                'text_hash': 'error',
                'status': 'error',
                'category': 'system_error',
                'llm_id': None,
                'latency_ms': int(latency_ms),
                'pre_score': None,
                'safety_score': None,
                'source_ip': '0.0.0.0',
                'token_count': 0,
                'error_message': error
            }
            
            self.logs.append(error_entry)
            logger.debug(f"Logged error for request {request_id}")
            
        except Exception as e:
            logger.error(f"Failed to log error for {request_id}: {e}")
    
    async def get_logs(self, logs_request) -> Dict[str, Any]:
        """Получение логов"""
        try:
            # Фильтрация логов
            filtered_logs = self.logs.copy()
            
            # Фильтр по времени
            if logs_request.start_time:
                filtered_logs = [log for log in filtered_logs if log['ts'] >= logs_request.start_time]
            
            if logs_request.end_time:
                filtered_logs = [log for log in filtered_logs if log['ts'] <= logs_request.end_time]
            
            # Фильтр по статусу
            if logs_request.status:
                filtered_logs = [log for log in filtered_logs if log['status'] == logs_request.status]
            
            # Фильтр по категории
            if logs_request.category:
                filtered_logs = [log for log in filtered_logs if log.get('category') == logs_request.category]
            
            # Фильтр по LLM ID
            if logs_request.llm_id:
                filtered_logs = [log for log in filtered_logs if log.get('llm_id') == logs_request.llm_id]
            
            # Сортировка по времени (новые сначала)
            filtered_logs.sort(key=lambda x: x['ts'], reverse=True)
            
            # Пагинация
            total = len(filtered_logs)
            start_idx = logs_request.offset
            end_idx = start_idx + logs_request.limit
            page_logs = filtered_logs[start_idx:end_idx]
            
            # Преобразование в нужный формат
            log_entries = []
            for log in page_logs:
                entry = {
                    'timestamp': log['ts'],
                    'request_id': log['request_id'],
                    'text_hash': log['text_hash'],
                    'status': log['status'],
                    'category': log.get('category'),
                    'llm_id': log.get('llm_id'),
                    'latency_ms': log['latency_ms'],
                    'pre_score': log.get('pre_score'),
                    'safety_score': log.get('safety_score'),
                    'source_ip': log['source_ip'],
                    'token_count': log.get('token_count', 0)
                }
                log_entries.append(entry)
            
            return {
                'logs': log_entries,
                'total': total,
                'has_more': end_idx < total
            }
            
        except Exception as e:
            logger.error(f"Failed to get logs: {e}")
            return {'logs': [], 'total': 0, 'has_more': False}
    
    async def get_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Получение статистики"""
        try:
            current_time = time.time()
            start_time = current_time - (hours * 3600)
            
            # Фильтруем логи за указанный период
            recent_logs = [log for log in self.logs if log['ts'] >= start_time]
            
            if not recent_logs:
                return {
                    'total_requests': 0,
                    'safe_requests': 0,
                    'unsafe_requests': 0,
                    'avg_latency_ms': 0.0,
                    'p95_latency_ms': 0.0,
                    'rps': 0.0,
                    'error_rate': 0.0,
                    'prefilter_hit_rate': 0.0
                }
            
            # Базовая статистика
            total_requests = len(recent_logs)
            safe_requests = len([log for log in recent_logs if log['status'] == 'safe'])
            unsafe_requests = len([log for log in recent_logs if log['status'] == 'unsafe'])
            error_requests = len([log for log in recent_logs if log['status'] == 'error'])
            
            # Статистика задержек
            latencies = [log['latency_ms'] for log in recent_logs if 'latency_ms' in log]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0
            
            # RPS
            time_period_seconds = hours * 3600
            rps = total_requests / time_period_seconds if time_period_seconds > 0 else 0.0
            
            # Error rate
            error_rate = (error_requests / total_requests * 100) if total_requests > 0 else 0.0
            
            # Pre-filter hit rate (приблизительно)
            # Считаем, что попадание в Pre-filter = быстрые safe ответы
            fast_safe = len([log for log in recent_logs 
                           if log['status'] == 'safe' and log.get('latency_ms', 0) < 100])
            prefilter_hit_rate = (fast_safe / total_requests * 100) if total_requests > 0 else 0.0
            
            return {
                'total_requests': total_requests,
                'safe_requests': safe_requests,
                'unsafe_requests': unsafe_requests,
                'avg_latency_ms': avg_latency,
                'p95_latency_ms': p95_latency,
                'rps': rps,
                'error_rate': error_rate,
                'prefilter_hit_rate': prefilter_hit_rate
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                'total_requests': 0,
                'safe_requests': 0,
                'unsafe_requests': 0,
                'avg_latency_ms': 0.0,
                'p95_latency_ms': 0.0,
                'rps': 0.0,
                'error_rate': 0.0,
                'prefilter_hit_rate': 0.0
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья хранилища логов"""
        try:
            # Проверяем, что можем записывать/читать
            test_log_count = len(self.logs)
            
            return {
                'status': 'healthy',
                'storage_type': 'in_memory_mock',
                'log_count': test_log_count,
                'max_capacity': self.max_logs
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def close(self):
        """Закрытие соединений"""
        logger.info("Mock LogWriter closed")
        # В реальной реализации здесь было бы закрытие соединения с ClickHouse


# Alias для совместимости
LogWriter = MockLogWriter


class ClickHouseLogWriter:
    """Реальная реализация с ClickHouse (для будущего использования)"""
    
    def __init__(self, clickhouse_url: str = "http://localhost:8123"):
        self.clickhouse_url = clickhouse_url
        self.client = None
        # TODO: Implement ClickHouse client initialization
        
    async def _init_connection(self):
        """Инициализация соединения с ClickHouse"""
        # TODO: Implement ClickHouse connection
        pass
        
    async def _ensure_table_exists(self):
        """Создание таблицы если не существует"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS filter_logs (
            ts DateTime64(3) CODEC(DoubleDelta, LZ4),
            request_id UUID,
            text_hash FixedString(32),
            status LowCardinality(String),
            category LowCardinality(String),
            llm_id LowCardinality(String),
            latency_ms UInt16,
            pre_score Float32,
            safety_score Float32,
            source_ip IPv6,
            token_count UInt16
        ) ENGINE = MergeTree
        PARTITION BY toYYYYMM(ts)
        ORDER BY (ts, status);
        """
        # TODO: Execute SQL
        pass 