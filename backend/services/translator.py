"""
翻译服务模块（异步版）

优化点：
1. 使用 httpx.AsyncClient 替代 deep-translator 的同步 requests
2. 连接池复用，减少 TCP 握手开销
3. 保留常见短语快速返回
"""
import logging
import httpx

from backend.config import settings

logger = logging.getLogger(__name__)

# Google Translate 免费 API endpoint（和 deep-translator 用的一样）
_TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"


class TranslatorService:
    """异步翻译服务"""
    
    def __init__(self) -> None:
        self._source = settings.source_lang
        self._target = settings.target_lang
        self._client: httpx.AsyncClient | None = None
        
        # 常见短语快速返回
        self._common_phrases = {
            "yeah": "是的",
            "yes": "是",
            "no": "不",
            "okay": "好的",
            "ok": "好",
            "thank you": "谢谢",
            "thanks": "谢谢",
            "sorry": "抱歉",
            "excuse me": "不好意思",
            "hello": "你好",
            "hi": "嗨",
            "bye": "再见",
            "goodbye": "再见",
        }
    
    def _get_client(self) -> httpx.AsyncClient:
        """懒初始化 httpx 客户端（连接池复用）"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=10.0,
                limits=httpx.Limits(
                    max_connections=5,
                    max_keepalive_connections=3,
                ),
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/120.0.0.0 Safari/537.36",
                },
            )
        return self._client
    
    async def translate(self, text: str) -> str:
        """
        异步翻译文本
        
        Args:
            text: 待翻译文本
            
        Returns:
            翻译结果
        """
        if not text or not text.strip():
            return ""
        
        text = text.strip()
        
        # 常见短语直接返回
        text_lower = text.lower()
        if text_lower in self._common_phrases:
            return self._common_phrases[text_lower]
        
        try:
            client = self._get_client()
            
            response = await client.get(
                _TRANSLATE_URL,
                params={
                    "client": "gtx",
                    "sl": self._source,
                    "tl": self._target,
                    "dt": "t",
                    "q": text,
                },
            )
            response.raise_for_status()
            
            # 解析 Google Translate 返回的 JSON
            # 格式: [[[翻译结果, 原文, ...]]]
            result_data = response.json()
            
            if result_data and result_data[0]:
                translated_parts = []
                for part in result_data[0]:
                    if part[0]:
                        translated_parts.append(part[0])
                result = "".join(translated_parts)
                return result if result else ""
            
            return ""
        except Exception as e:
            logger.error("Translation failed: %s", e)
            return text  # 翻译失败返回原文
    
    async def close(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# 全局实例
translator_service = TranslatorService()
