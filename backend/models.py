"""
数据模型定义
"""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RecognitionRecord(BaseModel):
    """识别记录模型"""
    
    timestamp: int = Field(..., description="时间戳(毫秒)")
    text: str = Field(..., description="识别文本")
    session_key: str = Field(..., description="会话标识")
    lang: str = Field(..., description="语言 (en/zh)")
    
    @classmethod
    def create(cls, text: str, session_key: str, lang: str = "en") -> "RecognitionRecord":
        """创建识别记录"""
        return cls(
            timestamp=int(datetime.now().timestamp() * 1000),
            text=text,
            session_key=session_key,
            lang=lang,
        )


class TranslationRecord(BaseModel):
    """翻译记录模型"""
    
    timestamp: int = Field(..., description="时间戳(毫秒)")
    original_text: str = Field(..., description="原文")
    translated_text: str = Field(..., description="译文")
    session_key: str = Field(..., description="会话标识")
    
    @classmethod
    def create(cls, original: str, translated: str, session_key: str) -> "TranslationRecord":
        """创建翻译记录"""
        return cls(
            timestamp=int(datetime.now().timestamp() * 1000),
            original_text=original,
            translated_text=translated,
            session_key=session_key,
        )


class TranslationResponse(BaseModel):
    """翻译响应"""
    
    success: bool
    message: str
    data: dict[str, Any] | None = None


class QueryRequest(BaseModel):
    """查询请求"""
    
    session_key: str = Field(..., description="会话key,格式: 2025_12_24_1770702412632")
    start_index: int = Field(0, ge=0, description="起始索引")
    count: int = Field(50, ge=1, le=500, description="获取数量")


class QueryResponse(BaseModel):
    """查询响应"""
    
    session_key: str
    total: int
    records: list[TranslationRecord]
