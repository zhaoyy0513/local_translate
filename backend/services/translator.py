"""
翻译服务模块
"""
import logging

from deep_translator import GoogleTranslator

from backend.config import settings

logger = logging.getLogger(__name__)


class TranslatorService:
    """翻译服务封装"""
    
    def __init__(self) -> None:
        self._translator = GoogleTranslator(
            source=settings.source_lang,
            target=settings.target_lang,
        )
        # 上下文记忆 (最近 5 条)
        self._context = []
    
    def translate(self, text: str) -> str:
        """
        翻译文本
        
        Args:
            text: 待翻译文本
            
        Returns:
            翻译结果
            
        Raises:
            Exception: 翻译失败
        """
        if not text or not text.strip():
            return ""
        
        text = text.strip()
        
        # 处理常见短语 (避免翻译不准)
        common_phrases = {
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
        
        text_lower = text.lower()
        if text_lower in common_phrases:
            return common_phrases[text_lower]
        
        try:
            # 如果有上下文,拼接一起翻译
            if self._context and len(text.split()) < 5:
                context_text = " ".join(self._context[-2:]) + " " + text
                result = self._translator.translate(context_text)
                # 只取最后一部分
                if result:
                    parts = result.split()
                    result = " ".join(parts[-len(text.split()):])
            else:
                result = self._translator.translate(text)
            
            # 更新上下文
            self._context.append(text)
            if len(self._context) > 5:
                self._context.pop(0)
            
            return result if result else ""
        except Exception as e:
            logger.error("Translation failed: %s", e)
            # 如果翻译失败,返回原文
            return text


# 全局实例
translator_service = TranslatorService()
