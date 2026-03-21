"""
FastAPI 服务端使用示例
展示如何使用异步 TTS 接口处理并发请求
"""

import sys
from pathlib import Path

# 强制导入项目目录中的 gsv_tts 模块
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from gsv_tts import TTS
import uuid
import os

app = FastAPI(title="GSV-TTS 异步 API", version="1.0")

base_dir = Path(__file__).parent
models_dir = base_dir / "WebUI" / "models"
output_dir = base_dir / "output"
output_dir.mkdir(exist_ok=True)

tts: Optional[TTS] = None

class TTSSingleRequest(BaseModel):
    text: str
    speaker_audio: str
    prompt_audio: str
    prompt_text: str
    top_k: int = 5
    top_p: float = 0.9
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    noise_scale: float = 0.5
    speed: float = 1.0

class TTSBatchRequest(BaseModel):
    texts: List[str]
    speaker_audio: str
    prompt_audio: str
    prompt_text: str
    top_k: int = 5
    top_p: float = 0.9
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    noise_scale: float = 0.5
    speed: float = 1.0

@app.on_event("startup")
async def startup_event():
    global tts
    print("🚀 正在加载 TTS 模型...")
    tts = TTS(
        models_dir=str(models_dir),
        gpt_cache=[(1, 512), (4, 512), (8, 512)],
        sovits_cache=[50],
    )
    print("✅ TTS 模型加载完成！")
    
    # 调试信息：检查 TTS 对象的方法
    print(f"调试信息：")
    print(f"  - hasattr(tts, 'infer_async'): {hasattr(tts, 'infer_async')}")
    print(f"  - hasattr(tts, 'infer_batched_async'): {hasattr(tts, 'infer_batched_async')}")
    print(f"  - TTS 类位置: {type(tts).__module__}")

@app.get("/")
async def root():
    return {"message": "GSV-TTS 异步 API 服务已启动", "docs": "/docs"}

@app.post("/tts/single")
async def tts_single(request: TTSSingleRequest):
    """单个 TTS 请求的异步接口"""
    try:
        audio_clip = await tts.infer_async(
            spk_audio_path=request.speaker_audio,
            prompt_audio_path=request.prompt_audio,
            prompt_audio_text=request.prompt_text,
            text=request.text,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
            noise_scale=request.noise_scale,
            speed=request.speed,
        )
        
        output_filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
        output_path = output_dir / output_filename
        audio_clip.save(str(output_path))
        
        return {
            "success": True,
            "audio_len": audio_clip.audio_len_s,
            "filename": output_filename,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/batch")
async def tts_batch(request: TTSBatchRequest):
    """批量 TTS 请求的异步接口"""
    try:
        audio_clips = await tts.infer_batched_async(
            spk_audio_paths=request.speaker_audio,
            prompt_audio_paths=request.prompt_audio,
            prompt_audio_texts=request.prompt_text,
            texts=request.texts,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
            noise_scale=request.noise_scale,
            speed=request.speed,
        )
        
        filenames = []
        for clip in audio_clips:
            filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
            output_path = output_dir / filename
            clip.save(str(output_path))
            filenames.append(filename)
        
        return {
            "success": True,
            "count": len(audio_clips),
            "filenames": filenames,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """获取生成的音频文件"""
    file_path = output_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件未找到")
    return FileResponse(file_path, media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
