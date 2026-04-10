"""
个人化应用 API
核心目标：追求简单、功能全的交互体验
调用方式：支持 infer_stream、infer_batched 两种推理模式，中间无需复杂的调度逻辑
"""

import sys
from pathlib import Path
import asyncio
import base64
import json
import logging
import os
import tempfile
import uuid
from typing import Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field

from gsv_tts import TTS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="GSV-TTS 个人化应用 API",
    description="简单、功能全的TTS API，支持流式和批量两种推理模式",
    version="1.0"
)

models_dir = project_root / "API" / "models"
output_dir = project_root / "output"
output_dir.mkdir(exist_ok=True)

tts: Optional[TTS] = None
asr = None
temp_dir = tempfile.mkdtemp(prefix="gsv_tts_personal_")


def is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def resolve_audio_path(path: str) -> str:
    if is_url(path):
        return path
    p = Path(path)
    if p.is_absolute():
        return path
    resolved = project_root / path
    if resolved.exists():
        return str(resolved)
    return path


async def download_audio(url: str) -> str:
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
    global asr
    if asr is None:
        raise HTTPException(status_code=500, detail="ASR模型未启用，请提供 prompt_text")
    
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


class TTSStreamRequest(BaseModel):
    text: str = Field(..., description="要合成的文本")
    speaker_audio: str = Field(..., description="说话人参考音频路径或URL")
    prompt_audio: str = Field(..., description="提示音频路径或URL")
    prompt_text: Optional[str] = Field(None, description="提示音频文本，为空时自动ASR识别")
    
    is_cut_text: bool = Field(True, description="是否按标点切分文本")
    cut_minlen: int = Field(10, description="文本切分最小长度")
    cut_mute: float = Field(0.2, description="切分后的静音时长(秒)")
    
    stream_mode: str = Field("token", description="流式模式: token 或 sentence")
    stream_chunk: int = Field(25, description="token模式下每次生成的token数")
    overlap_len: int = Field(10, description="重叠长度，用于平滑拼接")
    boost_first_chunk: bool = Field(True, description="是否加速首个chunk生成")
    
    top_k: int = Field(15, description="GPT采样top_k")
    top_p: float = Field(1.0, description="GPT采样top_p")
    temperature: float = Field(1.0, description="GPT采样温度")
    repetition_penalty: float = Field(1.35, description="重复惩罚")
    noise_scale: float = Field(0.5, description="噪声强度")
    speed: float = Field(1.0, description="语速")


class TTSBatchedRequest(BaseModel):
    texts: list[str] = Field(..., description="要合成的文本列表")
    speaker_audio: str = Field(..., description="说话人参考音频路径或URL")
    prompt_audio: str = Field(..., description="提示音频路径或URL")
    prompt_text: Optional[str] = Field(None, description="提示音频文本，为空时自动ASR识别")
    
    is_cut_text: bool = Field(True, description="是否按标点切分文本")
    cut_minlen: int = Field(10, description="文本切分最小长度")
    cut_mute: float = Field(0.2, description="切分后的静音时长(秒)")
    
    return_subtitles: bool = Field(False, description="是否返回字幕时间戳")
    
    top_k: int = Field(15, description="GPT采样top_k")
    top_p: float = Field(1.0, description="GPT采样top_p")
    temperature: float = Field(1.0, description="GPT采样温度")
    repetition_penalty: float = Field(1.35, description="重复惩罚")
    noise_scale: float = Field(0.5, description="噪声强度")
    speed: float = Field(1.0, description="语速")


@app.on_event("startup")
async def startup_event():
    global tts, asr
    print("正在加载 TTS 模型...")
    
    max_cache_len = 1024
    batch_sizes = [1, 4, 8]
    cache_lens = []
    length = 512
    while length <= max_cache_len:
        cache_lens.append(length)
        length *= 2
    gpt_cache = [(b, c) for b in batch_sizes for c in cache_lens]
    
    tts = TTS(
        models_dir=str(models_dir),
        gpt_cache=gpt_cache,
        sovits_cache=[50],
    )
    print("TTS 模型加载完成！")
    
    use_asr = os.environ.get("USE_ASR", "true").lower() == "true"
    if use_asr:
        try:
            import torch
            from huggingface_hub import snapshot_download
            
            local_model_path = models_dir / "qwen3_asr"
            repo_id = "Qwen/Qwen3-ASR-0.6B"
            
            if not (local_model_path.exists() and (local_model_path / "config.json").exists()):
                print(f"本地未找到ASR模型，正在下载: {repo_id}")
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_model_path),
                    local_dir_use_symlinks=False,
                )
                print("ASR模型下载完成！")
            
            from qwen_asr import Qwen3ASRModel
            print("正在加载 ASR 模型...")
            asr = Qwen3ASRModel.from_pretrained(
                str(local_model_path),
                dtype=torch.bfloat16,
                device_map="cuda:0",
                local_files_only=True
            )
            print("ASR 模型加载完成！")
        except Exception as e:
            print(f"ASR 模型加载失败: {e}")
            print("提示：如果没有提供 prompt_text，请求将会失败")
            asr = None
    else:
        print("ASR 模型已禁用")


@app.get("/")
async def root():
    return {
        "message": "GSV-TTS 个人化应用 API",
        "version": "1.0",
        "endpoints": {
            "stream": "/tts/stream - 流式推理 (SSE)",
            "batched": "/tts/batched - 批量推理",
            "audio": "/audio/{filename} - 获取音频文件"
        },
        "features": {
            "url_support": True,
            "auto_asr": asr is not None
        }
    }


@app.post("/tts/stream")
async def tts_stream(request: TTSStreamRequest):
    """
    流式推理API - 使用SSE实时推送音频片段
    
    适用场景：
    - 实时对话
    - 长文本生成
    - 需要低延迟响应
    
    返回格式：Server-Sent Events (SSE)
    - event: audio - 音频片段(base64编码)
    - event: subtitle - 字幕信息
    - event: done - 生成完成
    - event: error - 错误信息
    """
    try:
        speaker_audio = resolve_audio_path(request.speaker_audio)
        prompt_audio = resolve_audio_path(request.prompt_audio)
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
        
        final_prompt_text = prompt_text
        final_speaker_audio = speaker_audio
        final_prompt_audio = prompt_audio
        
        async def generate():
            try:
                loop = asyncio.get_running_loop()
                
                def stream_infer():
                    return list(tts.infer_stream(
                        spk_audio_path=final_speaker_audio,
                        prompt_audio_path=final_prompt_audio,
                        prompt_audio_text=final_prompt_text,
                        text=request.text,
                        is_cut_text=request.is_cut_text,
                        cut_minlen=request.cut_minlen,
                        cut_mute=request.cut_mute,
                        stream_mode=request.stream_mode,
                        stream_chunk=request.stream_chunk,
                        overlap_len=request.overlap_len,
                        boost_first_chunk=request.boost_first_chunk,
                        top_k=request.top_k,
                        top_p=request.top_p,
                        temperature=request.temperature,
                        repetition_penalty=request.repetition_penalty,
                        noise_scale=request.noise_scale,
                        speed=request.speed,
                        debug=False,
                    ))
                
                clips = await loop.run_in_executor(None, stream_infer)
                
                total_len = 0
                for clip in clips:
                    audio_bytes = clip.audio_data.tobytes()
                    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                    
                    total_len += len(clip.audio_data)
                    
                    chunk_data = {
                        "audio": audio_b64,
                        "sample_rate": clip.samplerate,
                        "duration": clip.audio_len_s,
                        "subtitles": clip.subtitles,
                        "text": clip.orig_text
                    }
                    
                    yield f"event: audio\ndata: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                
                yield f"event: done\ndata: {json.dumps({'total_duration': total_len / 32000}, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                logging.error(f"流式推理错误: {e}")
                yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/batched")
async def tts_batched(request: TTSBatchedRequest):
    """
    批量推理API - 一次请求生成多个音频
    
    适用场景：
    - 批量生成
    - 离线处理
    - 不需要实时响应
    
    返回格式：JSON
    - success: 是否成功
    - count: 生成的音频数量
    - filenames: 生成的音频文件名列表
    - subtitles: 字幕信息(可选)
    """
    try:
        speaker_audio = resolve_audio_path(request.speaker_audio)
        prompt_audio = resolve_audio_path(request.prompt_audio)
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
            return_subtitles=request.return_subtitles,
            is_cut_text=request.is_cut_text,
            cut_minlen=request.cut_minlen,
            cut_mute=request.cut_mute,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
            noise_scale=request.noise_scale,
            speed=request.speed,
        )
        
        filenames = []
        subtitles_list = []
        
        for clip in audio_clips:
            filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
            output_path = output_dir / filename
            clip.save(str(output_path))
            filenames.append(filename)
            
            if request.return_subtitles and clip.subtitles:
                subtitles_list.append(clip.subtitles)
        
        result = {
            "success": True,
            "count": len(audio_clips),
            "filenames": filenames,
            "prompt_text_used": prompt_text,
        }
        
        if request.return_subtitles:
            result["subtitles"] = subtitles_list
        
        return result
        
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
