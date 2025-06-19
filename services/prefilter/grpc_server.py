"""
gRPC сервер для Pre-Filter сервиса
"""

import asyncio
import logging
import grpc
from concurrent import futures
from typing import Dict, Any
import sys
from pathlib import Path

# Добавляем путь к proto файлам
sys.path.append(str(Path(__file__).parent.parent.parent))

from services.prefilter.service import PreFilterService

# TODO: Сгенерировать protobuf файлы
# from proto import filter_pb2_grpc, filter_pb2

logger = logging.getLogger(__name__)


class PreFilterGRPCService:
    """gRPC реализация Pre-Filter сервиса"""
    
    def __init__(self, service: PreFilterService):
        self.service = service
    
    async def Predict(self, request, context):
        """Обработка gRPC запроса предсказания"""
        try:
            # Валидация запроса
            if not request.text:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Текст не может быть пустым")
                return None
            
            # Предсказание
            result = await self.service.predict(request.text)
            
            # Формирование ответа
            response = filter_pb2.PreFilterResponse(
                safe=result['safe'],
                uncertain=result['uncertain'],
                score=result['score'],
                request_id=request.request_id
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Ошибка обработки gRPC запроса: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Внутренняя ошибка: {str(e)}")
            return None


class PreFilterServer:
    """HTTP/REST сервер для разработки (до генерации gRPC)"""
    
    def __init__(self, port: int = 50051):
        self.port = port
        self.service = None
        self.server = None
    
    async def start(self):
        """Запуск сервера"""
        try:
            # Инициализация сервиса
            self.service = PreFilterService()
            
            # Создание gRPC сервера (заглушка)
            # TODO: Реализовать после генерации proto файлов
            logger.info(f"Pre-Filter сервер запускается на порту {self.port}")
            
            # Временная REST-заглушка для разработки
            await self._start_rest_server()
            
        except Exception as e:
            logger.error(f"Ошибка запуска сервера: {e}")
            raise
    
    async def _start_rest_server(self):
        """Временный REST сервер для разработки"""
        from aiohttp import web, web_request
        import json
        
        async def predict_handler(request: web_request.Request):
            try:
                data = await request.json()
                text = data.get('text', '')
                request_id = data.get('request_id', '')
                
                result = await self.service.predict(text)
                
                response = {
                    'safe': result['safe'],
                    'uncertain': result['uncertain'], 
                    'score': result['score'],
                    'request_id': request_id,
                    'inference_ms': result['inference_ms']
                }
                
                return web.json_response(response)
                
            except Exception as e:
                logger.error(f"Ошибка REST handler: {e}")
                return web.json_response(
                    {'error': str(e)}, 
                    status=500
                )
        
        async def health_handler(request):
            health = self.service.health_check()
            status = 200 if health['status'] == 'healthy' else 503
            return web.json_response(health, status=status)
        
        async def stats_handler(request):
            stats = self.service.get_stats()
            return web.json_response(stats)
        
        # Создание приложения
        app = web.Application()
        app.router.add_post('/predict', predict_handler)
        app.router.add_get('/health', health_handler)
        app.router.add_get('/stats', stats_handler)
        
        # Запуск сервера
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        
        logger.info(f"REST сервер запущен на http://0.0.0.0:{self.port}")
        logger.info("Доступные эндпоинты:")
        logger.info("  POST /predict - предсказание")
        logger.info("  GET /health - проверка здоровья")
        logger.info("  GET /stats - статистика")
        
        # Держим сервер работающим
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Получен сигнал остановки")
        finally:
            await runner.cleanup()
    
    async def stop(self):
        """Остановка сервера"""
        if self.service:
            self.service.close()
        
        if self.server:
            self.server.stop(0)
        
        logger.info("Pre-Filter сервер остановлен")


async def main():
    """Основная функция запуска сервера"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pre-Filter gRPC Server')
    parser.add_argument('--port', type=int, default=50051, help='Порт сервера')
    parser.add_argument('--model-path', type=str, 
                       default='models/artifacts/pre_filter.onnx',
                       help='Путь к ONNX модели')
    args = parser.parse_args()
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Создание и запуск сервера
    server = PreFilterServer(port=args.port)
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Получен Ctrl+C, остановка сервера...")
    finally:
        await server.stop()


if __name__ == '__main__':
    asyncio.run(main()) 