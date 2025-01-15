import os
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from loguru import logger

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler

app = FastAPI()

# Initialize the video sampler at startup
args = parse_args()
models_root_path = Path(args.model_base)
if not models_root_path.exists():
    raise ValueError(f"`models_root` not exists: {models_root_path}")

# Create temporary directory for video storage
TEMP_DIR = Path("temp_videos")
TEMP_DIR.mkdir(exist_ok=True)

# Initialize the sampler
hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
args = hunyuan_video_sampler.args  # Get updated args

class VideoRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    height: Optional[int] = args.video_size[0]
    width: Optional[int] = args.video_size[1]
    video_length: Optional[int] = args.video_length
    seed: Optional[int] = args.seed
    infer_steps: Optional[int] = args.infer_steps
    guidance_scale: Optional[float] = args.cfg_scale
    num_videos: Optional[int] = 1
    flow_shift: Optional[float] = args.flow_shift
    embedded_guidance_scale: Optional[float] = args.embedded_cfg_scale

@app.post("/generate")
async def generate_video(request: VideoRequest):
    try:
        # Generate the video
        outputs = hunyuan_video_sampler.predict(
            prompt=request.prompt,
            height=request.height,
            width=request.width,
            video_length=request.video_length,
            seed=request.seed,
            negative_prompt=request.negative_prompt,
            infer_steps=request.infer_steps,
            guidance_scale=request.guidance_scale,
            num_videos_per_prompt=request.num_videos,
            flow_shift=request.flow_shift,
            embedded_guidance_scale=request.embedded_guidance_scale
        )
        
        samples = outputs['samples']
        
        # Save the video temporarily
        time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
        filename = f"{time_flag}_seed{outputs['seeds'][0]}_{outputs['prompts'][0][:50].replace('/','')}.mp4"
        temp_path = TEMP_DIR / filename
        
        # Save the first video (in case multiple were generated)
        save_videos_grid(samples[0].unsqueeze(0), str(temp_path), fps=24)
        
        # Return the video file and schedule cleanup
        # return FileResponse(
        #     path=temp_path,
        #     filename=filename,
        #     media_type="video/mp4",
        #     background=cleanup_file(temp_path)
        # )

    except Exception as e:
        logger.error(f"Error generating video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def cleanup_file(file_path: Path):
    """Delete the temporary file after it has been sent"""
    try:
        os.remove(file_path)
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up the temporary directory on shutdown"""
    try:
        shutil.rmtree(TEMP_DIR)
    except Exception as e:
        logger.error(f"Error cleaning up temp directory: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)