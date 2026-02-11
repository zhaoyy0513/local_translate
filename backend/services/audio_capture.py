"""
音频捕获与识别服务
"""
import asyncio
import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pyaudio
from faster_whisper import WhisperModel

from backend.config import settings
from backend.models import RecognitionRecord
from backend.services.redis_client import redis_client
from backend.services.translator import translator_service

# WebSocket 连接管理
active_connections: list = []

# 线程池用于处理异步操作
executor = ThreadPoolExecutor(max_workers=2)

logger = logging.getLogger(__name__)


class AudioCaptureService:
    """音频捕捉与实时翻译服务"""
    
    def __init__(self) -> None:
        self._is_running = False
        self._session_key: str | None = None
        self._audio_thread: threading.Thread | None = None
        self._translate_thread: threading.Thread | None = None
        self._model: WhisperModel | None = None
        self._device_index: int | None = None
        
        # 翻译队列 (异步处理)
        self._translate_queue: queue.Queue = queue.Queue()
        
        # PyAudio 配置
        self._format = pyaudio.paInt16
        self._channels = 2  # 立体声
        self._rate = settings.sample_rate
        self._chunk_size = int(self._rate * settings.chunk_duration)
    
    def _detect_audio_device(self) -> tuple[int | None, int]:
        """
        获取音频设备
        
        Returns:
            (device_index, sample_rate)
        """
        # 如果配置了设备索引,直接使用
        if settings.audio_device_index is not None:
            logger.info(
                "Using configured audio device index: %d (rate: %d Hz)",
                settings.audio_device_index,
                settings.sample_rate,
            )
            return settings.audio_device_index, settings.sample_rate
        
        audio = pyaudio.PyAudio()
        try:
            # 查找包含关键字的设备 (优先 WASAPI)
            keywords = [
                "stereo mix",
                "立体声混音",
                "loopback",
                "what u hear",
                "wave out mix",
            ]
            
            candidates = []
            
            for i in range(audio.get_device_count()):
                info = audio.get_device_info_by_index(i)
                
                # 必须是输入设备
                if info["maxInputChannels"] < 1:
                    continue
                
                device_name = info["name"].lower()
                host_api = audio.get_host_api_info_by_index(info["hostApi"])["name"]
                
                # 匹配关键字
                for keyword in keywords:
                    if keyword in device_name:
                        # 优先选择 WASAPI 设备
                        priority = 1 if "WASAPI" in host_api else 2
                        candidates.append((priority, i, info))
                        break
            
            if candidates:
                # 按优先级排序
                candidates.sort(key=lambda x: x[0])
                _, device_index, info = candidates[0]
                
                sample_rate = int(info["defaultSampleRate"])
                
                logger.info(
                    "Auto-detected audio device: [%d] %s (Rate: %d Hz)",
                    device_index,
                    info["name"],
                    sample_rate,
                )
                
                return device_index, sample_rate
            
            # 如果没找到,使用默认设备
            logger.warning(
                "Stereo Mix not found, using default input device. "
                "You may need to enable 'Stereo Mix' in Windows Sound settings."
            )
            return None, settings.sample_rate
            
        finally:
            audio.terminate()
    
    def _load_model(self) -> None:
        """延迟加载 Whisper 模型"""
        if self._model is None:
            logger.info("Loading Whisper model: %s", settings.whisper_model)
            self._model = WhisperModel(
                settings.whisper_model,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
            )
            logger.info("Model loaded successfully")
    
    def _translate_worker(self) -> None:
        """翻译工作线程 (异步处理翻译)"""
        logger.info("Translation worker started")
        
        while self._is_running or not self._translate_queue.empty():
            try:
                # 从队列获取任务 (超时 1 秒)
                item = self._translate_queue.get(timeout=1)
                
                text = item["text"]
                timestamp = item["timestamp"]
                
                # 翻译
                try:
                    translated = translator_service.translate(text)
                    logger.info("Translated: %s -> %s", text, translated)
                    
                    # 保存中文记录
                    zh_record = RecognitionRecord.create(
                        text=translated,
                        session_key=self._session_key,
                        lang="zh",
                    )
                    zh_record.timestamp = timestamp  # 使用同一时间戳
                    
                    self._run_async_in_thread(
                        redis_client.save_recognition(zh_record)
                    )
                    
                    # WebSocket 推送
                    self._run_async_in_thread(
                        self._broadcast({
                            "type": "translation",
                            "lang": "zh",
                            "text": translated,
                            "timestamp": timestamp,
                        })
                    )
                    
                except Exception as e:
                    logger.error("Translation failed: %s", e)
                
                self._translate_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Translation worker error: %s", e)
        
        logger.info("Translation worker stopped")
    
    def _run_async_in_thread(self, coro):
        """在新线程中运行异步函数"""
        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        
        future = executor.submit(run)
        return future
    
    async def _broadcast(self, message: dict) -> None:
        """广播消息到所有 WebSocket 连接"""
        import json
        disconnected = []
        
        for connection in active_connections:
            try:
                await connection.send_text(json.dumps(message, ensure_ascii=False))
            except Exception as e:
                logger.error("Failed to send message: %s", e)
                disconnected.append(connection)
        
        # 清理断开的连接
        for conn in disconnected:
            active_connections.remove(conn)
    
    def _capture_and_process(self) -> None:
        """音频捕捉线程主函数"""
        audio = pyaudio.PyAudio()
        
        try:
            # 打开音频流 (使用检测到的设备)
            stream_params = {
                "format": self._format,
                "channels": self._channels,
                "rate": self._rate,
                "input": True,
                "frames_per_buffer": 1024,
            }
            
            if self._device_index is not None:
                stream_params["input_device_index"] = self._device_index
                logger.info("Using audio device index: %d", self._device_index)
            else:
                logger.info("Using default audio device")
            
            stream = audio.open(**stream_params)
            
            logger.info("Audio capture started")
            
            while self._is_running:
                # 读取音频块
                frames = []
                for _ in range(0, int(self._rate / 1024 * settings.chunk_duration)):
                    if not self._is_running:
                        break
                    data = stream.read(1024, exception_on_overflow=False)
                    frames.append(data)
                
                if not frames:
                    continue
                
                # 转换为 numpy 数组
                audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
                
                # 如果是立体声,转为单声道 (取平均)
                if self._channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1).astype(np.int16)
                
                audio_float = audio_data.astype(np.float32) / 32768.0
                
                # 简单 VAD: 检测音量
                volume = np.abs(audio_float).mean()
                logger.debug("Audio volume: %.4f", volume)
                if volume < 0.005:  # 静音阈值 (降低)
                    continue
                
                # 重采样到 16000 Hz (Whisper 需要)
                if self._rate != 16000:
                    # 简单的重采样: 按比例取样
                    ratio = 16000 / self._rate
                    new_length = int(len(audio_float) * ratio)
                    indices = np.linspace(0, len(audio_float) - 1, new_length).astype(int)
                    audio_float = audio_float[indices]
                
                # 语音识别 (简化参数,提高稳定性)
                try:
                    segments, info = self._model.transcribe(
                        audio_float,
                        language="en",
                        beam_size=5,
                        vad_filter=True,
                    )
                    
                    for segment in segments:
                        text = segment.text.strip()
                        if not text:
                            continue
                        
                        logger.info("Recognized: %s", text)
                        
                        # 保存英文记录
                        try:
                            en_record = RecognitionRecord.create(
                                text=text,
                                session_key=self._session_key,
                                lang="en",
                            )
                            
                            # 保存到 Redis
                            self._run_async_in_thread(
                                redis_client.save_recognition(en_record)
                            )
                            
                            # WebSocket 推送
                            self._run_async_in_thread(
                                self._broadcast({
                                    "type": "recognition",
                                    "lang": "en",
                                    "text": text,
                                    "timestamp": en_record.timestamp,
                                })
                            )
                            
                            # 加入翻译队列 (异步处理,不阻塞识别)
                            if settings.enable_translation:
                                self._translate_queue.put({
                                    "text": text,
                                    "timestamp": en_record.timestamp,
                                })
                            
                        except Exception as e:
                            logger.error("Save English failed: %s", e)
                
                except Exception as e:
                    logger.error("Transcription failed: %s", e)
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logger.error("Audio capture error: %s", e)
            self._is_running = False
        finally:
            audio.terminate()
            logger.info("Audio capture stopped")
    
    def start_capture(self) -> str:
        """
        开始音频捕获
        
        Returns:
            session_key
            
        Raises:
            RuntimeError: 已在运行中
        """
        if self._is_running:
            raise RuntimeError("Capture already running")
        
        # 生成 session_key (格式: 2026_02_11_14_30_1707621643000)
        now = datetime.now()
        timestamp_ms = int(now.timestamp() * 1000)
        self._session_key = now.strftime(f"%Y_%m_%d_%H_%M_{timestamp_ms}")
        
        # 检测音频设备
        self._device_index, self._rate = self._detect_audio_device()
        logger.info("Using sample rate: %d Hz", self._rate)
        
        # 加载模型
        self._load_model()
        
        # 启动音频捕捉线程
        self._is_running = True
        self._audio_thread = threading.Thread(
            target=self._capture_and_process,
            daemon=True,
        )
        self._audio_thread.start()
        
        # 启动翻译线程
        if settings.enable_translation:
            self._translate_thread = threading.Thread(
                target=self._translate_worker,
                daemon=True,
            )
            self._translate_thread.start()
        
        logger.info("Started capture with session: %s", self._session_key)
        return self._session_key
    
    def stop_capture(self) -> None:
        """停止音频捕捉"""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # 等待音频线程
        if self._audio_thread:
            self._audio_thread.join(timeout=5)
        
        # 等待翻译队列清空
        if self._translate_thread:
            self._translate_queue.join()  # 等待所有任务完成
            self._translate_thread.join(timeout=10)
        
        logger.info("Stopped capture for session: %s", self._session_key)
        self._session_key = None
    
    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._is_running
    
    @property
    def current_session(self) -> str | None:
        """当前会话key"""
        return self._session_key


# 全局实例
audio_service = AudioCaptureService()
