"""
音频捕获与识别服务（sherpa-onnx 流式版）

核心特性：
- sherpa-onnx Zipformer 流式模型：准确度接近 Whisper，真实时流式
- 每 100ms 出 partial result（边说边出字）
- 内置 endpoint detection（自动检测句子结束）
- WebSocket 推送 partial + final 两种消息
"""
import asyncio
import json
import logging
import os
import threading
import tarfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pyaudio

from backend.config import settings
from backend.models import RecognitionRecord
from backend.services.redis_client import redis_client
from backend.services.translator import translator_service

# WebSocket 连接管理
active_connections: list = []

logger = logging.getLogger(__name__)

# sherpa-onnx 模型配置
MODEL_NAME = "sherpa-onnx-streaming-zipformer-en-2023-06-26"
MODEL_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{MODEL_NAME}.tar.bz2"
# 备用镜像 (HuggingFace)
MODEL_URL_HF = f"https://huggingface.co/csukuangfj/{MODEL_NAME}/resolve/main/encoder-epoch-99-avg-1-chunk-16-left-128.onnx"
MODEL_DIR = Path(__file__).parent.parent.parent / "models"


def _ensure_model() -> Path:
    """确保 sherpa-onnx 模型已下载，返回模型目录"""
    model_path = MODEL_DIR / MODEL_NAME
    
    # 检查模型文件是否存在
    encoder = model_path / "encoder-epoch-99-avg-1-chunk-16-left-128.onnx"
    if encoder.exists():
        return model_path
    
    # 检查是否有其他 sherpa 模型
    if MODEL_DIR.exists():
        for d in MODEL_DIR.iterdir():
            if d.is_dir() and d.name.startswith("sherpa-onnx-streaming"):
                # 找到 encoder 文件
                encoders = list(d.glob("encoder*.onnx"))
                if encoders:
                    logger.info("Found existing model: %s", d.name)
                    return d
    
    # 下载模型
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = MODEL_DIR / f"{MODEL_NAME}.tar.bz2"
    
    logger.info("=" * 60)
    logger.info("Downloading sherpa-onnx model: %s", MODEL_NAME)
    logger.info("This is a one-time download (~300MB), please wait...")
    logger.info("URL: %s", MODEL_URL)
    logger.info("=" * 60)
    
    import requests
    
    try:
        response = requests.get(MODEL_URL, stream=True, timeout=30)
        response.raise_for_status()
    except Exception as e:
        logger.warning("GitHub download failed (%s), trying alternative...", e)
        # 如果 GitHub 下载失败，提示用户手动下载
        raise RuntimeError(
            f"模型下载失败。请手动下载模型：\n"
            f"1. 访问 https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models\n"
            f"2. 下载 {MODEL_NAME}.tar.bz2\n"
            f"3. 解压到 {MODEL_DIR}\n"
        ) from e
    
    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    
    with open(tar_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total and downloaded % (20 * 1024 * 1024) < 65536:
                logger.info("Download: %.0f%% (%dMB / %dMB)",
                    downloaded / total * 100,
                    downloaded // (1024 * 1024),
                    total // (1024 * 1024),
                )
    
    logger.info("Extracting model...")
    with tarfile.open(tar_path, "r:bz2") as tf:
        tf.extractall(MODEL_DIR)
    
    tar_path.unlink()
    logger.info("Model ready: %s", model_path)
    
    return model_path


class AudioCaptureService:
    """音频捕捉与实时翻译服务（sherpa-onnx 流式）"""
    
    def __init__(self) -> None:
        self._is_running = False
        self._session_key: str | None = None
        self._audio_thread: threading.Thread | None = None
        self._device_index: int | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        
        # PyAudio 配置
        self._format = pyaudio.paInt16
        self._channels = 2  # 立体声
        self._rate = settings.sample_rate
    
    def _detect_audio_device(self) -> tuple[int | None, int]:
        """获取音频设备"""
        if settings.audio_device_index is not None:
            logger.info(
                "Using configured audio device index: %d (rate: %d Hz)",
                settings.audio_device_index,
                settings.sample_rate,
            )
            return settings.audio_device_index, settings.sample_rate
        
        audio = pyaudio.PyAudio()
        try:
            keywords = [
                "stereo mix", "立体声混音", "loopback",
                "what u hear", "wave out mix",
            ]
            candidates = []
            
            for i in range(audio.get_device_count()):
                info = audio.get_device_info_by_index(i)
                if info["maxInputChannels"] < 1:
                    continue
                device_name = info["name"].lower()
                host_api = audio.get_host_api_info_by_index(info["hostApi"])["name"]
                for keyword in keywords:
                    if keyword in device_name:
                        priority = 1 if "WASAPI" in host_api else 2
                        candidates.append((priority, i, info))
                        break
            
            if candidates:
                candidates.sort(key=lambda x: x[0])
                _, device_index, info = candidates[0]
                sample_rate = int(info["defaultSampleRate"])
                logger.info(
                    "Auto-detected: [%d] %s (Rate: %d Hz)",
                    device_index, info["name"], sample_rate,
                )
                return device_index, sample_rate
            
            logger.warning("Stereo Mix not found, using default input device.")
            return None, settings.sample_rate
        finally:
            audio.terminate()
    
    def _schedule_async(self, coro) -> None:
        """将协程投递到主 event loop"""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self._loop)
    
    async def _save_and_broadcast(self, record: RecognitionRecord, msg_type: str) -> None:
        """并发 Redis 保存 + WebSocket 推送"""
        message = {
            "type": msg_type,
            "lang": record.lang,
            "text": record.text,
            "timestamp": record.timestamp,
        }
        await asyncio.gather(
            redis_client.save_recognition(record),
            self._broadcast(message),
        )
    
    async def _broadcast_partial(self, text: str) -> None:
        """推送 partial result（不保存到 Redis）"""
        message = {
            "type": "partial",
            "lang": "en",
            "text": text,
            "timestamp": int(datetime.now().timestamp() * 1000),
        }
        await self._broadcast(message)
    
    async def _translate_and_save(self, text: str, timestamp: int) -> None:
        """异步翻译 + 保存 + 推送"""
        try:
            translated = await translator_service.translate(text)
            if not translated:
                return
            
            logger.info("Translated: %s -> %s", text, translated)
            
            zh_record = RecognitionRecord.create(
                text=translated,
                session_key=self._session_key,
                lang="zh",
            )
            zh_record.timestamp = timestamp
            
            await self._save_and_broadcast(zh_record, "translation")
        except Exception as e:
            logger.error("Translation failed: %s", e)
    
    async def _broadcast(self, message: dict) -> None:
        """广播消息到所有 WebSocket 连接"""
        data = json.dumps(message, ensure_ascii=False)
        disconnected = []
        for connection in active_connections:
            try:
                await connection.send_text(data)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            if conn in active_connections:
                active_connections.remove(conn)
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """将全大写的模型输出转为正常大小写"""
        if not text:
            return text
        # 全小写后，首字母大写
        text = text.lower().strip()
        if text:
            text = text[0].upper() + text[1:]
        return text
    
    def _capture_and_process(self) -> None:
        """
        音频捕捉线程 — sherpa-onnx 流式识别
        
        每 100ms 读取音频 → 喂给 recognizer →
        partial result 实时推送 →
        endpoint 检测到句子结束 → final result 保存+翻译
        """
        import sherpa_onnx
        
        # 加载模型
        model_path = _ensure_model()
        logger.info("Loading sherpa-onnx model from: %s", model_path)
        
        # 查找模型文件
        encoders = sorted(model_path.glob("encoder*.onnx"))
        decoders = sorted(model_path.glob("decoder*.onnx"))
        joiners = sorted(model_path.glob("joiner*.onnx"))
        tokens = model_path / "tokens.txt"
        
        if not encoders or not decoders or not joiners or not tokens.exists():
            raise RuntimeError(
                f"模型文件不完整。需要 encoder/decoder/joiner .onnx + tokens.txt\n"
                f"模型目录: {model_path}"
            )
        
        # 创建 recognizer
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(tokens),
            encoder=str(encoders[0]),
            decoder=str(decoders[0]),
            joiner=str(joiners[0]),
            num_threads=4,
            sample_rate=16000,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,   # 纯静音 2.4s → endpoint
            rule2_min_trailing_silence=0.8,   # 说了话后 0.8s 静音 → endpoint
            rule3_min_utterance_length=20.0,  # 超长句子 20s → endpoint
            decoding_method="greedy_search",
        )
        
        stream = recognizer.create_stream()
        
        logger.info("sherpa-onnx recognizer ready")
        
        audio = pyaudio.PyAudio()
        
        # 每次读取 100ms 音频
        vosk_rate = 16000
        read_duration = 0.1  # 100ms
        read_frames = int(self._rate * read_duration)
        
        # 重采样参数
        need_resample = self._rate != vosk_rate
        if need_resample:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(self._rate, vosk_rate)
            resample_up = vosk_rate // g
            resample_down = self._rate // g
            logger.info(
                "Resampling: %d -> %d Hz (up=%d, down=%d)",
                self._rate, vosk_rate, resample_up, resample_down,
            )
        
        try:
            stream_params = {
                "format": self._format,
                "channels": self._channels,
                "rate": self._rate,
                "input": True,
                "frames_per_buffer": read_frames,
            }
            
            if self._device_index is not None:
                stream_params["input_device_index"] = self._device_index
                logger.info("Using audio device index: %d", self._device_index)
            
            audio_stream = audio.open(**stream_params)
            logger.info(
                "Streaming started (read=%dms, rate=%d->%d Hz)",
                int(read_duration * 1000), self._rate, vosk_rate,
            )
            
            last_partial = ""
            
            while self._is_running:
                # 读取 100ms 音频
                data = audio_stream.read(read_frames, exception_on_overflow=False)
                
                # 转 numpy int16
                audio_int16 = np.frombuffer(data, dtype=np.int16)
                
                # 立体声转单声道
                if self._channels == 2:
                    audio_int16 = audio_int16.reshape(-1, 2).mean(axis=1).astype(np.int16)
                
                # 转 float32 (sherpa-onnx 需要 [-1, 1] 范围的 float)
                audio_float = audio_int16.astype(np.float32) / 32768.0
                
                # 重采样到 16kHz
                if need_resample:
                    audio_float = resample_poly(
                        audio_float, resample_up, resample_down,
                    ).astype(np.float32)
                
                # 喂给 recognizer
                stream.accept_waveform(vosk_rate, audio_float)
                
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)
                
                # 获取当前识别结果
                current_text = self._normalize_text(recognizer.get_result(stream))
                
                is_endpoint = recognizer.is_endpoint(stream)
                
                if is_endpoint and current_text:
                    # === Final: 句子说完了 ===
                    logger.info("Final: %s", current_text)
                    
                    en_record = RecognitionRecord.create(
                        text=current_text,
                        session_key=self._session_key,
                        lang="en",
                    )
                    
                    self._schedule_async(
                        self._save_and_broadcast(en_record, "recognition")
                    )
                    
                    if settings.enable_translation:
                        self._schedule_async(
                            self._translate_and_save(current_text, en_record.timestamp)
                        )
                    
                    recognizer.reset(stream)
                    last_partial = ""
                    
                elif is_endpoint and not current_text:
                    # 空 endpoint，重置
                    recognizer.reset(stream)
                    last_partial = ""
                    
                elif current_text and current_text != last_partial:
                    # === Partial: 正在说话，实时更新 ===
                    last_partial = current_text
                    self._schedule_async(
                        self._broadcast_partial(current_text)
                    )
            
            # 处理剩余
            tail_text = recognizer.get_result(stream).strip()
            if tail_text:
                logger.info("Final (flush): %s", tail_text)
                en_record = RecognitionRecord.create(
                    text=tail_text,
                    session_key=self._session_key,
                    lang="en",
                )
                self._schedule_async(
                    self._save_and_broadcast(en_record, "recognition")
                )
                if settings.enable_translation:
                    self._schedule_async(
                        self._translate_and_save(tail_text, en_record.timestamp)
                    )
            
            audio_stream.stop_stream()
            audio_stream.close()
            
        except Exception as e:
            logger.error("Audio capture error: %s", e)
            self._is_running = False
        finally:
            audio.terminate()
            logger.info("Audio capture stopped")
    
    def start_capture(self) -> str:
        """开始音频捕获"""
        if self._is_running:
            raise RuntimeError("Capture already running")
        
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
            logger.warning("No running event loop")
        
        now = datetime.now()
        timestamp_ms = int(now.timestamp() * 1000)
        self._session_key = now.strftime(f"%Y_%m_%d_%H_%M_{timestamp_ms}")
        
        self._device_index, self._rate = self._detect_audio_device()
        logger.info("Using sample rate: %d Hz", self._rate)
        
        self._is_running = True
        self._audio_thread = threading.Thread(
            target=self._capture_and_process,
            daemon=True,
        )
        self._audio_thread.start()
        
        logger.info("Started capture with session: %s", self._session_key)
        return self._session_key
    
    def stop_capture(self) -> None:
        """停止音频捕捉"""
        if not self._is_running:
            return
        
        self._is_running = False
        
        if self._audio_thread:
            self._audio_thread.join(timeout=5)
        
        logger.info("Stopped capture for session: %s", self._session_key)
        self._session_key = None
    
    @property
    def is_running(self) -> bool:
        return self._is_running
    
    @property
    def current_session(self) -> str | None:
        return self._session_key


# 全局实例
audio_service = AudioCaptureService()
