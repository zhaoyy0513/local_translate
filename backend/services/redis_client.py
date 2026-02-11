"""
Redis 操作服务
"""
import json
from datetime import datetime

import redis.asyncio as aioredis

from backend.config import settings
from backend.models import RecognitionRecord, TranslationRecord


class RedisClient:
    """Redis 客户端封装"""
    
    def __init__(self) -> None:
        self._client: aioredis.Redis | None = None
    
    async def connect(self) -> None:
        """建立 Redis 连接"""
        self._client = await aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
    
    async def disconnect(self) -> None:
        """关闭 Redis 连接"""
        if self._client:
            await self._client.close()
    
    async def save_recognition(self, record: RecognitionRecord) -> None:
        """
        保存识别记录到 Redis Stream
        
        Args:
            record: 识别记录
            
        Raises:
            redis.RedisError: Redis 操作失败
        """
        if not self._client:
            raise RuntimeError("Redis client not connected")
        
        # 根据语言分开存储
        stream_key = f"session:{record.session_key}:{record.lang}"
        
        await self._client.xadd(
            stream_key,
            {
                "timestamp": str(record.timestamp),
                "text": record.text,
            },
        )
        
        # 设置过期时间 (7天)
        await self._client.expire(stream_key, 7 * 24 * 60 * 60)
    
    async def query_records(
        self,
        session_key: str,
        lang: str = "en",
        start_index: int = 0,
        count: int = 50,
    ) -> tuple[int, list[RecognitionRecord]]:
        """
        查询识别记录
        
        Args:
            session_key: 会话标识
            lang: 语言 (en/zh)
            start_index: 起始索引
            count: 获取数量
            
        Returns:
            (总数, 记录列表)
            
        Raises:
            redis.RedisError: Redis 操作失败
        """
        if not self._client:
            raise RuntimeError("Redis client not connected")
        
        stream_key = f"session:{session_key}:{lang}"
        
        # 获取 Stream 长度
        total = await self._client.xlen(stream_key)
        
        if total == 0:
            return 0, []
        
        entries = await self._client.xrange(
            stream_key,
            min="-",
            max="+",
            count=start_index + count,
        )
        
        entries = entries[start_index:]
        
        records = []
        for entry_id, data in entries:
            records.append(
                RecognitionRecord(
                    timestamp=int(data["timestamp"]),
                    text=data["text"],
                    session_key=session_key,
                    lang=lang,
                )
            )
        
        return total, records
    
    async def list_sessions(self) -> list[str]:
        """
        列出所有会话
        
        Returns:
            会话key列表
        """
        if not self._client:
            raise RuntimeError("Redis client not connected")
        
        # 使用 SCAN 遍历 key (避免 KEYS 阻塞)
        sessions_set = set()
        async for key in self._client.scan_iter(match="session:*:en", count=100):
            # 提取 session_key (session:2026_02_11_14_30:en -> 2026_02_11_14_30)
            session_key = key.replace("session:", "").replace(":en", "")
            sessions_set.add(session_key)
        
        return sorted(list(sessions_set), reverse=True)


# 全局实例
redis_client = RedisClient()
