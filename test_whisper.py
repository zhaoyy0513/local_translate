"""
Whisper 识别测试脚本
直接测试音频捕获 + Whisper 识别
"""
import numpy as np
import pyaudio
from faster_whisper import WhisperModel
import time

# 配置
DEVICE_INDEX = 24  # WASAPI 立体声混音
SAMPLE_RATE = 48000
CHANNELS = 2
DURATION = 5  # 录制 5 秒

print("=" * 60)
print("Whisper 识别测试")
print("=" * 60)

# 加载模型
print("\n[1/3] 加载 Whisper 模型 (base)...")
model = WhisperModel("base", device="cpu", compute_type="int8")
print("✅ 模型加载完成")

# 录制音频
print(f"\n[2/3] 开始录制 {DURATION} 秒音频...")
print("请播放 YouTube 英文视频!\n")

audio = pyaudio.PyAudio()

stream = audio.open(
    format=pyaudio.paInt16,
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=1024,
)

frames = []
start_time = time.time()

while time.time() - start_time < DURATION:
    data = stream.read(1024, exception_on_overflow=False)
    frames.append(data)
    
    # 显示进度
    elapsed = time.time() - start_time
    progress = int(elapsed / DURATION * 50)
    bar = "█" * progress + "░" * (50 - progress)
    print(f"\r录制中: [{bar}] {elapsed:.1f}s", end="")

print("\n✅ 录制完成")

stream.stop_stream()
stream.close()
audio.terminate()

# 处理音频
print("\n[3/3] Whisper 识别...")

audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)

# 立体声转单声道
if CHANNELS == 2:
    audio_data = audio_data.reshape(-1, 2).mean(axis=1).astype(np.int16)

# 转换为浮点数
audio_float = audio_data.astype(np.float32) / 32768.0

# 检查音频数据
print(f"\n音频信息 (原始):")
print(f"  - 采样数: {len(audio_float)}")
print(f"  - 时长: {len(audio_float) / SAMPLE_RATE:.2f} 秒")
print(f"  - 最大值: {np.max(np.abs(audio_float)):.4f}")
print(f"  - 平均音量: {np.mean(np.abs(audio_float)):.4f}")

# 重采样到 16000 Hz (Whisper 需要)
if SAMPLE_RATE != 16000:
    print(f"\n重采样: {SAMPLE_RATE} Hz -> 16000 Hz")
    ratio = 16000 / SAMPLE_RATE
    new_length = int(len(audio_float) * ratio)
    indices = np.linspace(0, len(audio_float) - 1, new_length).astype(int)
    audio_float = audio_float[indices]
    print(f"  - 新采样数: {len(audio_float)}")
    print(f"  - 新时长: {len(audio_float) / 16000:.2f} 秒")

# 检查是否有有效音频
if np.max(np.abs(audio_float)) < 0.01:
    print("\n❌ 音频数据几乎为静音! 检查:")
    print("   - YouTube 是否正在播放")
    print("   - 系统音量是否静音")
    print("   - 是否选择了正确的音频设备")
else:
    print(f"\n音频数据正常,开始识别...")
    
    # Whisper 识别
    segments, info = model.transcribe(
        audio_float,
        language="en",
        beam_size=5,
        vad_filter=True,
    )
    
    print(f"\n检测到语言: {info.language} (概率: {info.language_probability:.2%})")
    print(f"\n识别结果:")
    print("-" * 40)
    
    found_text = False
    for segment in segments:
        text = segment.text.strip()
        if text:
            found_text = True
            print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {text}")
    
    if not found_text:
        print("(无识别结果)")
        print("\n可能原因:")
        print("  1. 音频中没有英语语音")
        print("  2. 音量太小")
        print("  3. 音频质量差")
        print("  4. VAD 过滤掉了语音")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
