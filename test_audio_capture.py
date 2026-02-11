"""
音频捕获测试脚本
验证是否能正确捕获系统音频
"""
import pyaudio
import numpy as np
import time

# 配置
CHUNK = 1024
CHANNELS = 2
DURATION = 10  # 测试 10 秒

print("=" * 60)
print("音频捕捉测试")
print("=" * 60)

# 检测立体声混音设备
audio = pyaudio.PyAudio()
device_index = None
sample_rate = 16000

keywords = ["stereo mix", "立体声混音", "loopback", "what u hear"]

# 优先选择 WASAPI 设备
candidates = []

for i in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(i)
    
    if info['maxInputChannels'] < 1:
        continue
    
    device_name = info['name'].lower()
    host_api = audio.get_host_api_info_by_index(info['hostApi'])['name']
    
    for keyword in keywords:
        if keyword in device_name:
            priority = 1 if "WASAPI" in host_api else 2
            candidates.append((priority, i, info))
            break

# 尝试设备 24 (WASAPI 立体声混音)
device_index = 24
info = audio.get_device_info_by_index(device_index)
sample_rate = int(info['defaultSampleRate'])
print(f"\n✅ 使用音频设备: [{device_index}] {info['name']}")
print(f"采样率: {sample_rate} Hz")
print(f"Host API: {audio.get_host_api_info_by_index(info['hostApi'])['name']}")

audio.terminate()

RATE = sample_rate

# 开始捕获
print(f"\n开始捕获音频 ({DURATION} 秒)...")
print("请播放 YouTube 视频或音乐...\n")

audio = pyaudio.PyAudio()

stream_params = {
    "format": pyaudio.paInt16,
    "channels": CHANNELS,
    "rate": RATE,
    "input": True,
    "frames_per_buffer": CHUNK,
}

if device_index is not None:
    stream_params["input_device_index"] = device_index

stream = audio.open(**stream_params)

try:
    start_time = time.time()
    max_volume = 0
    
    while time.time() - start_time < DURATION:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # 计算音量
        volume = np.abs(audio_data).mean()
        max_volume = max(max_volume, volume)
        
        # 显示进度条
        elapsed = time.time() - start_time
        progress = int(elapsed / DURATION * 50)
        bar = "█" * progress + "░" * (50 - progress)
        
        # 音量指示器
        volume_bar = "▓" * min(int(volume / 500), 20)
        
        print(f"\r[{bar}] {elapsed:.1f}s  音量: {volume_bar:<20} ({volume:.0f})", end="")
    
    print(f"\n\n最大音量: {max_volume:.0f}")
    
    if max_volume < 100:
        print("❌ 音量过低! 可能的原因:")
        print("   - 未播放音频")
        print("   - 立体声混音未启用或未设为默认设备")
        print("   - 系统音量静音")
    elif max_volume < 1000:
        print("⚠️  音量偏低,但可检测到音频")
    else:
        print("✅ 音频捕获正常!")
    
except KeyboardInterrupt:
    print("\n\n测试中断")
except Exception as e:
    print(f"\n\n❌ 错误: {e}")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
