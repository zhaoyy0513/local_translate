"""
音频设备检测脚本
用于查看可用的音频输入设备
"""
import pyaudio

audio = pyaudio.PyAudio()

print("=" * 60)
print("可用音频设备列表:")
print("=" * 60)

for i in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(i)
    
    # 只显示输入设备
    if info['maxInputChannels'] > 0:
        print(f"\n[设备 {i}]")
        print(f"  名称: {info['name']}")
        print(f"  输入通道: {info['maxInputChannels']}")
        print(f"  采样率: {int(info['defaultSampleRate'])} Hz")
        print(f"  Host API: {audio.get_host_api_info_by_index(info['hostApi'])['name']}")

audio.terminate()

print("\n" + "=" * 60)
print("提示: 如果看不到 'Realtek HD Audio 2nd output' 或类似的")
print("播放设备,说明需要使用虚拟音频线缆 (如 VB-Audio Cable)")
print("=" * 60)
