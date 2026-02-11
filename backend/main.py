"""
FastAPI 主应用
"""
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from backend.config import settings
from backend.models import QueryRequest, TranslationResponse, RecognitionRecord
from backend.services.audio_capture import audio_service, active_connections
from backend.services.redis_client import redis_client

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("Connecting to Redis...")
    await redis_client.connect()
    logger.info("Application started")
    
    yield
    
    # 关闭时
    logger.info("Stopping audio capture...")
    audio_service.stop_capture()
    logger.info("Disconnecting from Redis...")
    await redis_client.disconnect()
    logger.info("Application shutdown complete")


app = FastAPI(
    title="实时语音翻译 API",
    description="Teams 会议实时翻译服务",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """健康检查"""
    return {"status": "ok", "message": "实时语音翻译服务正在运行"}


@app.post("/api/translation/start", response_model=TranslationResponse)
async def start_translation():
    """开始实时翻译"""
    try:
        if audio_service.is_running:
            return TranslationResponse(
                success=False,
                message="翻译已在进行中",
                data={"session_key": audio_service.current_session},
            )
        
        session_key = audio_service.start_capture()
        
        return TranslationResponse(
            success=True,
            message="翻译已启动",
            data={"session_key": session_key},
        )
    
    except Exception as e:
        logger.error("Failed to start translation: %s", e)
        raise HTTPException(status_code=500, detail=f"启动失败: {str(e)}")


@app.post("/api/translation/stop", response_model=TranslationResponse)
async def stop_translation():
    """停止实时翻译"""
    try:
        if not audio_service.is_running:
            return TranslationResponse(
                success=False,
                message="当前没有运行中的翻译任务",
            )
        
        session_key = audio_service.current_session
        audio_service.stop_capture()
        
        return TranslationResponse(
            success=True,
            message="翻译已停止",
            data={"session_key": session_key},
        )
    
    except Exception as e:
        logger.error("Failed to stop translation: %s", e)
        raise HTTPException(status_code=500, detail=f"停止失败: {str(e)}")


@app.get("/api/translation/status", response_model=TranslationResponse)
async def get_status():
    """获取翻译状态"""
    return TranslationResponse(
        success=True,
        message="ok",
        data={
            "is_running": audio_service.is_running,
            "current_session": audio_service.current_session,
        },
    )


@app.get("/api/records/{session_key}/{lang}")
async def query_records(
    session_key: str,
    lang: str,
    start_index: int = 0,
    count: int = 50,
):
    """查询识别记录"""
    try:
        if lang not in ["en", "zh"]:
            raise HTTPException(status_code=400, detail="语言必须为 en 或 zh")
        
        total, records = await redis_client.query_records(
            session_key=session_key,
            lang=lang,
            start_index=start_index,
            count=count,
        )
        
        return {
            "success": True,
            "session_key": session_key,
            "lang": lang,
            "total": total,
            "records": [r.dict() for r in records],
        }
    
    except Exception as e:
        logger.error("Failed to query records: %s", e)
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.get("/api/sessions", response_model=TranslationResponse)
async def list_sessions():
    """列出所有会话"""
    try:
        sessions = await redis_client.list_sessions()
        return TranslationResponse(
            success=True,
            message="ok",
            data={"sessions": sessions},
        )
    
    except Exception as e:
        logger.error("Failed to list sessions: %s", e)
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {str(e)}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """实时推送 WebSocket"""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info("WebSocket connected. Total: %d", len(active_connections))
    
    try:
        while True:
            # 保持连接活跃
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info("WebSocket disconnected. Total: %d", len(active_connections))


@app.get("/api/config")
async def get_config():
    """获取配置信息"""
    return {
        "enable_translation": settings.enable_translation,
        "source_lang": settings.source_lang,
        "target_lang": settings.target_lang,
    }


@app.get("/api/export/{session_key}")
async def export_session(
    session_key: str,
    format: str = "txt",
    lang: str = "both",
):
    """
    导出会话记录
    
    Args:
        session_key: 会话标识
        format: 导出格式 (txt/json)
        lang: 语言选择 (en/zh/both)
    """
    try:
        en_records: list = []
        zh_records: list = []
        
        if lang in ["en", "both"]:
            _, en_records = await redis_client.query_records(
                session_key=session_key, lang="en", count=10000
            )
        if lang in ["zh", "both"]:
            _, zh_records = await redis_client.query_records(
                session_key=session_key, lang="zh", count=10000
            )
        
        if format == "json":
            if lang == "both":
                data = {
                    "session_key": session_key,
                    "exported_at": datetime.now().isoformat(),
                    "english": [r.dict() for r in en_records],
                    "chinese": [r.dict() for r in zh_records],
                }
            else:
                records = en_records if lang == "en" else zh_records
                data = {
                    "session_key": session_key,
                    "exported_at": datetime.now().isoformat(),
                    "lang": lang,
                    "records": [r.dict() for r in records],
                }
            
            filename = f"{session_key}_{lang}.json"
            return Response(
                content=json.dumps(data, ensure_ascii=False, indent=2),
                media_type="application/json",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        else:
            # TXT 格式
            lines = []
            lines.append(f"会话: {session_key}")
            lines.append(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lang_label = {"en": "English", "zh": "中文", "both": "English + 中文"}[lang]
            lines.append(f"语言: {lang_label}")
            lines.append("=" * 80)
            lines.append("")
            
            if lang == "both":
                # 双语: 按时间戳合并排序
                all_records = []
                for r in en_records:
                    all_records.append((r.timestamp, "EN", r.text))
                for r in zh_records:
                    all_records.append((r.timestamp, "ZH", r.text))
                all_records.sort(key=lambda x: x[0])
                
                for ts, tag, text in all_records:
                    time_str = datetime.fromtimestamp(ts / 1000).strftime("%H:%M:%S")
                    lines.append(f"[{time_str}] [{tag}] {text}")
            else:
                # 单语言: 按原始顺序
                records = en_records if lang == "en" else zh_records
                for r in records:
                    time_str = datetime.fromtimestamp(r.timestamp / 1000).strftime("%H:%M:%S")
                    lines.append(f"[{time_str}] {r.text}")
            
            filename = f"{session_key}_{lang}.txt"
            return Response(
                content="\n".join(lines),
                media_type="text/plain; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
    
    except Exception as e:
        logger.error("Failed to export session: %s", e)
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误"},
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
