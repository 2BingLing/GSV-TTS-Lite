import asyncio
import time
import sys
from pathlib import Path

base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))

from gsv_tts import TTS

async def main():
    print("=" * 60)
    print("同步 vs 异步 TTS 性能测试")
    print("=" * 60)
    
    models_dir = base_dir / "WebUI" / "models"
    
    tts = TTS(
        models_dir=str(models_dir),
        gpt_cache=[(1, 512), (4, 512), (8, 512)],
        sovits_cache=[50],
    )
    
    print("\n✅ TTS 模型初始化完成")
    
    test_texts = [
        "今天天气真好，适合出去散步。",
        "人工智能正在改变我们的生活。",
        "语音合成技术越来越成熟了。",
        "我喜欢学习新的编程技术。",
        "这个项目真的很有趣！",
    ]
    
    speaker_audio = base_dir / "examples" / "laffey.mp3"
    prompt_audio = base_dir / "examples" / "AnAn.ogg"
    prompt_text = "ちが……ちがう。レイア、貴様は間違っている。"
    
    if not speaker_audio.exists():
        print(f"\n❌ 找不到示例音频文件: {speaker_audio}")
        print("请确保 examples/ 目录下有 AnAn.ogg 和 laffey.mp3")
        return
    
    print(f"\n📝 测试配置:")
    print(f"   - 请求数量: {len(test_texts)}")
    print(f"   - 说话人音频: {speaker_audio.name}")
    print(f"   - 提示音频: {prompt_audio.name}")
    
    # 预热：避免模型加载时间影响测试结果
    print("\n🔥 正在预热模型...")
    warmup_result = tts.infer(
        spk_audio_path=str(speaker_audio),
        prompt_audio_path=str(prompt_audio),
        prompt_audio_text=prompt_text,
        text="预热测试"
    )
    print("✅ 预热完成！")
    
    print("\n" + "-" * 60)
    print("测试 1: 同步逐个处理 (Sequential)")
    print("-" * 60)
    
    start_sync = time.time()
    sync_results = []
    
    for i, text in enumerate(test_texts):
        print(f"  处理请求 {i+1}/{len(test_texts)}...")
        result = tts.infer(
            spk_audio_path=str(speaker_audio),
            prompt_audio_path=str(prompt_audio),
            prompt_audio_text=prompt_text,
            text=text
        )
        sync_results.append(result)
    
    sync_time = time.time() - start_sync
    print(f"\n✅ 同步完成！耗时: {sync_time:.2f}秒")
    
    print("\n" + "-" * 60)
    print("测试 2: 异步批处理 (Batched)")
    print("-" * 60)
    
    start_async = time.time()
    
    print(f"  使用 infer_batched_async 处理 {len(test_texts)} 个请求...")
    async_results = await tts.infer_batched_async(
        spk_audio_paths=str(speaker_audio),
        prompt_audio_paths=str(prompt_audio),
        prompt_audio_texts=prompt_text,
        texts=test_texts
    )
    
    async_time = time.time() - start_async
    print(f"\n✅ 异步完成！耗时: {async_time:.2f}秒")
    
    print("\n" + "=" * 60)
    print("📊 性能对比结果")
    print("=" * 60)
    print(f"同步耗时: {sync_time:.2f}秒")
    print(f"异步耗时: {async_time:.2f}秒")
    print(f"速度提升: {sync_time / async_time:.2f}x")
    print(f"节省时间: {(sync_time - async_time):.2f}秒")
    
    if async_time < sync_time:
        print("\n🎉 结论: 异步处理更快！")
    else:
        print("\n📝 结论: 对于少量请求，差异不明显")
    
    print("\n💡 提示: 当有多个并发请求时，异步优势更明显！")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
