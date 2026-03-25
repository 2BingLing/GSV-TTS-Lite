"""
FastAPI 服务端使用示例
展示如何使用异步 TTS 接口处理并发请求
支持外链音频URL和ASR自动识别
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from gsv_tts import TTS
import uuid
import os
import tempfile
import logging

app = FastAPI(title="GSV-TTS 异步 API", version="1.1")

models_dir = project_root / "API" / "models"
output_dir = project_root / "output"
output_dir.mkdir(exist_ok=True)

tts: Optional[TTS] = None
asr = None

temp_dir = tempfile.mkdtemp(prefix="gsv_tts_")


def is_url(path: str) -> bool:
    """检查是否为URL"""
    return path.startswith("http://") or path.startswith("https://")


async def download_audio(url: str) -> str:
    """下载音频URL到临时文件"""
    import httpx
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
    
    ext = ".wav"
    content_type = response.headers.get("content-type", "")
    if "mp3" in content_type or url.lower().endswith(".mp3"):
        ext = ".mp3"
    elif "ogg" in content_type or url.lower().endswith(".ogg"):
        ext = ".ogg"
    elif "flac" in content_type or url.lower().endswith(".flac"):
        ext = ".flac"
    
    temp_path = os.path.join(temp_dir, f"download_{uuid.uuid4().hex}{ext}")
    with open(temp_path, "wb") as f:
        f.write(response.content)
    
    logging.info(f"下载音频到: {temp_path}")
    return temp_path


def transcribe_audio(audio_path: str) -> str:
    """使用ASR识别音频文本"""
    global asr
    if asr is None:
        raise HTTPException(status_code=500, detail="ASR模型未启用，请设置 --use_asr 或提供 prompt_text")
    
    results = asr.transcribe(audio_path)
    if results and len(results) > 0:
        result = results[0]
        if hasattr(result, 'text'):
            text = result.text
        elif isinstance(result, dict):
            text = result.get("text", "")
        else:
            text = str(result)
        logging.info(f"ASR识别结果: {text}")
        return text
    return ""


class TTSSingleRequest(BaseModel):
    text: str
    speaker_audio: str
    prompt_audio: str
    prompt_text: Optional[str] = None
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
    prompt_text: Optional[str] = None
    top_k: int = 5
    top_p: float = 0.9
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    noise_scale: float = 0.5
    speed: float = 1.0


@app.on_event("startup")
async def startup_event():
    global tts, asr
    print("🚀 正在加载 TTS 模型...")
    tts = TTS(
        models_dir=str(models_dir),
        gpt_cache=[(1, 512), (4, 512), (8, 512)],
        sovits_cache=[50],
    )
    print("✅ TTS 模型加载完成！")
    
    use_asr = os.environ.get("USE_ASR", "true").lower() == "true"
    if use_asr:
        try:
            import torch
            from huggingface_hub import snapshot_download
            
            local_model_path = models_dir / "qwen3_asr"
            repo_id = "Qwen/Qwen3-ASR-0.6B"
            
            if not (local_model_path.exists() and (local_model_path / "config.json").exists()):
                print(f"⬇️ 本地未找到ASR模型，正在下载: {repo_id}")
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_model_path),
                    local_dir_use_symlinks=False,
                )
                print("✅ ASR模型下载完成！")
            
            from qwen_asr import Qwen3ASRModel
            print("🚀 正在加载 ASR 模型...")
            asr = Qwen3ASRModel.from_pretrained(
                str(local_model_path),
                dtype=torch.bfloat16,
                device_map="cuda:0",
                local_files_only=True
            )
            print("✅ ASR 模型加载完成！")
        except Exception as e:
            print(f"⚠️ ASR 模型加载失败: {e}")
            print("💡 提示：如果没有提供 prompt_text，请求将会失败")
            asr = None
    else:
        print("ℹ️ ASR 模型已禁用")


@app.get("/")
async def root():
    return {
        "message": "GSV-TTS 异步 API 服务已启动",
        "docs": "/docs",
        "features": {
            "url_support": True,
            "auto_asr": asr is not None
        }
    }


@app.post("/tts/single")
async def tts_single(request: TTSSingleRequest):
    """单个 TTS 请求的异步接口，支持外链音频和自动ASR"""
    try:
        speaker_audio = request.speaker_audio
        prompt_audio = request.prompt_audio
        prompt_text = request.prompt_text
        
        if is_url(speaker_audio):
            speaker_audio = await download_audio(speaker_audio)
        
        if is_url(prompt_audio):
            prompt_audio = await download_audio(prompt_audio)
        
        if prompt_text is None or prompt_text == "":
            prompt_text = transcribe_audio(prompt_audio)
            if not prompt_text:
                raise HTTPException(
                    status_code=400, 
                    detail="无法自动识别prompt_audio文本，请手动提供prompt_text"
                )
        
        audio_clip = await tts.infer_async(
            spk_audio_path=speaker_audio,
            prompt_audio_path=prompt_audio,
            prompt_audio_text=prompt_text,
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
            "prompt_text_used": prompt_text,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/batch")
async def tts_batch(request: TTSBatchRequest):
    """批量 TTS 请求的异步接口，支持外链音频和自动ASR"""
    try:
        speaker_audio = request.speaker_audio
        prompt_audio = request.prompt_audio
        prompt_text = request.prompt_text
        
        if is_url(speaker_audio):
            speaker_audio = await download_audio(speaker_audio)
        
        if is_url(prompt_audio):
            prompt_audio = await download_audio(prompt_audio)
        
        if prompt_text is None or prompt_text == "":
            prompt_text = transcribe_audio(prompt_audio)
            if not prompt_text:
                raise HTTPException(
                    status_code=400, 
                    detail="无法自动识别prompt_audio文本，请手动提供prompt_text"
                )
        
        audio_clips = await tts.infer_batched_async(
            spk_audio_paths=speaker_audio,
            prompt_audio_paths=prompt_audio,
            prompt_audio_texts=prompt_text,
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
            "prompt_text_used": prompt_text,
        }
    except HTTPException:
        raise
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
