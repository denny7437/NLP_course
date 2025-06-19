"""
Аутентификация и rate limiting для Gateway
"""

import time
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque

try:
    import redis.asyncio as redis
    REDIS_ASYNC_AVAILABLE = True
    REDIS_AVAILABLE = True
except ImportError:
    try:
        import redis
        REDIS_ASYNC_AVAILABLE = False
        REDIS_AVAILABLE = True
    except ImportError:
        REDIS_ASYNC_AVAILABLE = False
        REDIS_AVAILABLE = False
        redis = None
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Конфигурация JWT
SECRET_KEY = "your-secret-key-change-in-production"  # В продакшене загружать из переменных окружения
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()


class User(BaseModel):
    """Модель пользователя"""
    user_id: str
    username: str
    role: str = "viewer"  # viewer, analyst, admin
    is_active: bool = True


class TokenData(BaseModel):
    """Данные токена"""
    user_id: Optional[str] = None
    username: Optional[str] = None
    role: Optional[str] = None


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Создание JWT токена"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> TokenData:
    """Проверка JWT токена"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        username: str = payload.get("username")
        role: str = payload.get("role", "viewer")
        
        if user_id is None:
            raise JWTError("Invalid token payload")
        
        token_data = TokenData(user_id=user_id, username=username, role=role)
        return token_data
        
    except JWTError as e:
        logger.warning(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Получение текущего пользователя из токена"""
    token = credentials.credentials
    token_data = verify_token(token)
    
    # В реальном приложении здесь был бы запрос к базе данных
    # Для MVP используем заглушку
    user = User(
        user_id=token_data.user_id,
        username=token_data.username or token_data.user_id,
        role=token_data.role or "viewer",
        is_active=True
    )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return user


def get_current_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """Проверка прав администратора"""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


class RateLimiter:
    """Rate limiter с поддержкой Redis и in-memory fallback"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_client = None
        self.memory_store = defaultdict(lambda: deque())
        self.redis_url = redis_url
        
        # Лимиты по умолчанию (requests per minute)
        self.default_limits = {
            "viewer": 100,
            "analyst": 500, 
            "admin": 1000
        }
        
        # Лимиты по IP (для защиты от спама)
        self.ip_limit = 200  # requests per minute
        
    async def _init_redis(self):
        """Инициализация Redis клиента"""
        if self.redis_url and not self.redis_client and REDIS_AVAILABLE:
            try:
                if REDIS_ASYNC_AVAILABLE:
                    self.redis_client = redis.from_url(self.redis_url)
                    await self.redis_client.ping()
                else:
                    self.redis_client = redis.from_url(self.redis_url)
                    self.redis_client.ping()
                logger.info("Redis client initialized for rate limiting")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}, using in-memory rate limiting")
                self.redis_client = None
        elif not REDIS_AVAILABLE:
            logger.info("Redis not available, using in-memory rate limiting")
    
    async def check_rate_limit(self, user_id: str, client_ip: str, 
                              role: str = "viewer", multiplier: int = 1):
        """Проверка rate limit для пользователя и IP"""
        await self._init_redis()
        
        current_time = time.time()
        window_size = 60  # 1 минута
        
        # Проверка лимита пользователя
        user_limit = self.default_limits.get(role, self.default_limits["viewer"]) * multiplier
        await self._check_limit(f"user:{user_id}", user_limit, window_size, current_time)
        
        # Проверка лимита IP
        ip_limit = self.ip_limit * multiplier
        await self._check_limit(f"ip:{client_ip}", ip_limit, window_size, current_time)
    
    async def _check_limit(self, key: str, limit: int, window: int, current_time: float):
        """Проверка лимита для конкретного ключа"""
        if self.redis_client:
            await self._check_redis_limit(key, limit, window, current_time)
        else:
            await self._check_memory_limit(key, limit, window, current_time)
    
    async def _check_redis_limit(self, key: str, limit: int, window: int, current_time: float):
        """Проверка лимита через Redis"""
        try:
            if REDIS_ASYNC_AVAILABLE:
                pipe = self.redis_client.pipeline()
                
                # Удаляем старые записи
                pipe.zremrangebyscore(key, 0, current_time - window)
                
                # Добавляем текущий запрос
                pipe.zadd(key, {str(current_time): current_time})
                
                # Получаем количество запросов в окне
                pipe.zcard(key)
                
                # Устанавливаем TTL
                pipe.expire(key, window)
                
                results = await pipe.execute()
                request_count = results[2]  # результат zcard
            else:
                # Синхронный Redis
                pipe = self.redis_client.pipeline()
                
                # Удаляем старые записи
                pipe.zremrangebyscore(key, 0, current_time - window)
                
                # Добавляем текущий запрос
                pipe.zadd(key, {str(current_time): current_time})
                
                # Получаем количество запросов в окне
                pipe.zcard(key)
                
                # Устанавливаем TTL
                pipe.expire(key, window)
                
                results = pipe.execute()
                request_count = results[2]  # результат zcard
            
            if request_count > limit:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded: {request_count}/{limit} requests per minute"
                )
                
        except Exception as e:
            logger.error(f"Redis error in rate limiting: {e}")
            # Fallback к memory-based лимитингу
            await self._check_memory_limit(key, limit, window, current_time)
    
    async def _check_memory_limit(self, key: str, limit: int, window: int, current_time: float):
        """Проверка лимита через in-memory хранилище"""
        requests = self.memory_store[key]
        
        # Удаляем старые запросы
        while requests and requests[0] < current_time - window:
            requests.popleft()
        
        # Проверяем лимит
        if len(requests) >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {len(requests)}/{limit} requests per minute"
            )
        
        # Добавляем текущий запрос
        requests.append(current_time)
    
    async def get_rate_limit_status(self, user_id: str, role: str = "viewer") -> Dict[str, Any]:
        """Получение статуса rate limit для пользователя"""
        current_time = time.time()
        window_size = 60
        key = f"user:{user_id}"
        limit = self.default_limits.get(role, self.default_limits["viewer"])
        
        if self.redis_client and REDIS_AVAILABLE:
            try:
                if REDIS_ASYNC_AVAILABLE:
                    count = await self.redis_client.zcard(key)
                else:
                    count = self.redis_client.zcard(key)
                remaining = max(0, limit - count)
                reset_time = current_time + window_size
                
                return {
                    "limit": limit,
                    "remaining": remaining,
                    "reset": reset_time,
                    "window": window_size
                }
            except Exception:
                pass
        
        # Fallback к memory store
        requests = self.memory_store[key]
        count = len([r for r in requests if r > current_time - window_size])
        remaining = max(0, limit - count)
        
        return {
            "limit": limit,
            "remaining": remaining,
            "reset": current_time + window_size,
            "window": window_size
        }
    
    async def close(self):
        """Закрытие соединений"""
        if self.redis_client and REDIS_AVAILABLE:
            try:
                if REDIS_ASYNC_AVAILABLE:
                    await self.redis_client.close()
                else:
                    self.redis_client.close()
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")


# Инстанс глобального rate limiter
rate_limiter = RateLimiter()


# Утилиты для создания тестовых токенов (для разработки)
def create_test_token(user_id: str, username: str, role: str = "viewer") -> str:
    """Создание тестового токена для разработки"""
    return create_access_token({
        "sub": user_id,
        "username": username,
        "role": role
    })


# Пример токенов для тестирования
TEST_TOKENS = {
    "viewer": create_test_token("test_viewer", "viewer_user", "viewer"),
    "analyst": create_test_token("test_analyst", "analyst_user", "analyst"), 
    "admin": create_test_token("test_admin", "admin_user", "admin")
} 