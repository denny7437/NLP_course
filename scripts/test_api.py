#!/usr/bin/env python3
"""
Скрипт тестирования API Gateway
"""

import asyncio
import httpx
import json
import time
from typing import List, Dict, Any

# Тестовые токены из auth.py
TEST_TOKENS = {
    "viewer": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0X3ZpZXdlciIsInVzZXJuYW1lIjoidmlld2VyX3VzZXIiLCJyb2xlIjoidmlld2VyIiwiZXhwIjoxNzM1NzQ0ODM2fQ.example",
    "admin": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0X2FkbWluIiwidXNlcm5hbWUiOiJhZG1pbl91c2VyIiwicm9sZSI6ImFkbWluIiwiZXhwIjoxNzM1NzQ0ODM2fQ.example"
}


class APITester:
    """Тестер API Gateway"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)
        
    async def test_health_check(self) -> Dict[str, Any]:
        """Тест health check"""
        print("🔍 Тестирование health check...")
        
        try:
            response = await self.client.get("/v1/health")
            result = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None,
                "error": response.text if response.status_code != 200 else None
            }
            
            if result["success"]:
                print("✅ Health check успешен")
            else:
                print(f"❌ Health check неуспешен: {result['error']}")
                
            return result
            
        except Exception as e:
            print(f"❌ Ошибка health check: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_filter_single(self, token_type: str = "viewer") -> Dict[str, Any]:
        """Тест фильтрации одного текста"""
        print("🔍 Тестирование фильтрации одного текста...")
        
        # Создаем временный токен для тестирования
        from services.gateway.auth import create_test_token
        token = create_test_token("test_user", "test_user", token_type)
        
        headers = {"Authorization": f"Bearer {token}"}
        
        test_cases = [
            {
                "text": "Hello, how are you today?",
                "expected": "safe"
            },
            {
                "text": "I hate everyone and want to kill them all",
                "expected": "unsafe"
            },
            {
                "text": "This is a simple test message",
                "expected": "safe"
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            try:
                payload = {
                    "text": test_case["text"],
                    "llm_id": "test-model"
                }
                
                start_time = time.time()
                response = await self.client.post("/v1/filter", json=payload, headers=headers)
                end_time = time.time()
                
                result = {
                    "test_case": i,
                    "text": test_case["text"][:50] + "...",
                    "expected": test_case["expected"],
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "response_time_ms": round((end_time - start_time) * 1000, 2),
                    "data": response.json() if response.status_code == 200 else None,
                    "error": response.text if response.status_code != 200 else None
                }
                
                if result["success"]:
                    actual_status = result["data"]["status"]
                    result["prediction_correct"] = actual_status == test_case["expected"]
                    print(f"✅ Тест {i}: {actual_status} (ожидался {test_case['expected']}) - {result['response_time_ms']}мс")
                else:
                    print(f"❌ Тест {i} неуспешен: {result['error']}")
                
                results.append(result)
                
            except Exception as e:
                print(f"❌ Ошибка в тесте {i}: {e}")
                results.append({
                    "test_case": i,
                    "success": False,
                    "error": str(e)
                })
        
        return {"test_results": results}
    
    async def test_filter_batch(self, token_type: str = "viewer") -> Dict[str, Any]:
        """Тест батч фильтрации"""
        print("🔍 Тестирование батч фильтрации...")
        
        from services.gateway.auth import create_test_token
        token = create_test_token("test_user", "test_user", token_type)
        headers = {"Authorization": f"Bearer {token}"}
        
        batch_requests = [
            {"text": "Hello world", "llm_id": "test"},
            {"text": "I hate you all", "llm_id": "test"},
            {"text": "Nice weather today", "llm_id": "test"}
        ]
        
        try:
            payload = {"requests": batch_requests}
            
            start_time = time.time()
            response = await self.client.post("/v1/filter/batch", json=payload, headers=headers)
            end_time = time.time()
            
            result = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response_time_ms": round((end_time - start_time) * 1000, 2),
                "batch_size": len(batch_requests),
                "data": response.json() if response.status_code == 200 else None,
                "error": response.text if response.status_code != 200 else None
            }
            
            if result["success"]:
                responses = result["data"]["responses"]
                result["responses_count"] = len(responses)
                print(f"✅ Батч тест успешен: {len(responses)} ответов за {result['response_time_ms']}мс")
            else:
                print(f"❌ Батч тест неуспешен: {result['error']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Ошибка батч теста: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_logs_and_stats(self, token_type: str = "admin") -> Dict[str, Any]:
        """Тест получения логов и статистики"""
        print("🔍 Тестирование логов и статистики...")
        
        from services.gateway.auth import create_test_token
        token = create_test_token("test_admin", "test_admin", token_type)
        headers = {"Authorization": f"Bearer {token}"}
        
        results = {}
        
        # Тест получения логов
        try:
            response = await self.client.get("/v1/logs?limit=10", headers=headers)
            results["logs"] = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None,
                "error": response.text if response.status_code != 200 else None
            }
            
            if results["logs"]["success"]:
                log_count = len(results["logs"]["data"]["logs"])
                print(f"✅ Получено {log_count} записей логов")
            else:
                print(f"❌ Ошибка получения логов: {results['logs']['error']}")
                
        except Exception as e:
            results["logs"] = {"success": False, "error": str(e)}
            print(f"❌ Ошибка теста логов: {e}")
        
        # Тест получения статистики
        try:
            response = await self.client.get("/v1/stats?hours=1", headers=headers)
            results["stats"] = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None,
                "error": response.text if response.status_code != 200 else None
            }
            
            if results["stats"]["success"]:
                stats = results["stats"]["data"]
                print(f"✅ Статистика: {stats['total_requests']} запросов, {stats['avg_latency_ms']:.1f}мс средняя задержка")
            else:
                print(f"❌ Ошибка получения статистики: {results['stats']['error']}")
                
        except Exception as e:
            results["stats"] = {"success": False, "error": str(e)}
            print(f"❌ Ошибка теста статистики: {e}")
        
        return results
    
    async def test_rate_limiting(self, token_type: str = "viewer") -> Dict[str, Any]:
        """Тест rate limiting"""
        print("🔍 Тестирование rate limiting...")
        
        from services.gateway.auth import create_test_token
        token = create_test_token("test_rate_limit", "test_rate_limit", token_type)
        headers = {"Authorization": f"Bearer {token}"}
        
        # Быстрая серия запросов
        payload = {"text": "Test rate limiting", "llm_id": "test"}
        
        success_count = 0
        rate_limited_count = 0
        error_count = 0
        
        # Делаем 20 быстрых запросов
        for i in range(20):
            try:
                response = await self.client.post("/v1/filter", json=payload, headers=headers)
                
                if response.status_code == 200:
                    success_count += 1
                elif response.status_code == 429:  # Too Many Requests
                    rate_limited_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                error_count += 1
        
        result = {
            "total_requests": 20,
            "success_count": success_count,
            "rate_limited_count": rate_limited_count,
            "error_count": error_count,
            "rate_limiting_works": rate_limited_count > 0
        }
        
        if result["rate_limiting_works"]:
            print(f"✅ Rate limiting работает: {rate_limited_count} запросов заблокированы")
        else:
            print(f"⚠️ Rate limiting не сработал (возможно, лимиты слишком высокие)")
        
        return result
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Запуск всех тестов"""
        print("🚀 Запуск полного набора тестов API...")
        print("=" * 60)
        
        results = {}
        
        # 1. Health check
        results["health"] = await self.test_health_check()
        
        # 2. Одиночная фильтрация
        results["single_filter"] = await self.test_filter_single()
        
        # 3. Батч фильтрация
        results["batch_filter"] = await self.test_filter_batch()
        
        # 4. Логи и статистика
        results["logs_stats"] = await self.test_logs_and_stats()
        
        # 5. Rate limiting
        results["rate_limiting"] = await self.test_rate_limiting()
        
        print("=" * 60)
        print("📊 Результаты тестирования:")
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, test_result in results.items():
            if isinstance(test_result, dict) and "success" in test_result:
                total_tests += 1
                if test_result["success"]:
                    passed_tests += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            else:
                print(f"📋 {test_name}: COMPLETED")
        
        print(f"\n🎯 Итого: {passed_tests}/{total_tests} тестов прошли успешно")
        
        return results
    
    async def close(self):
        """Закрытие клиента"""
        await self.client.aclose()


async def main():
    """Главная функция"""
    tester = APITester()
    
    try:
        results = await tester.run_all_tests()
        
        # Сохраняем результаты в файл
        with open("test_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📁 Результаты сохранены в test_results.json")
        
    finally:
        await tester.close()


if __name__ == "__main__":
    asyncio.run(main()) 