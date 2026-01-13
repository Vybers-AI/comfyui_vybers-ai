from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import requests
import copy
import json
import os
import urllib.parse
import time
import logging
from typing import List, Optional
import shutil

# ---------------- 配置 ----------------
API_KEY = "12345"
COMFYUI_API_URL = "http://0.0.0.0:8188"  # ComfyUI API 地址
WORKFLOW_FILE = "/workspace/fdy/comfyui_vybers-ai/wan2.1-t2v-1_3B_api.json"  # 工作流文件
POLL_INTERVAL = 2  # 轮询间隔（秒）
MAX_WAIT_TIME = 300  # 最大等待时间（秒）
STATIC_DIR = "videos"  # 视频存储目录
SERVER_BASE_URL = "http://server_ip:9003"  # 对外访问地址

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建存储目录
os.makedirs(STATIC_DIR, exist_ok=True)

# ---------------- 读取工作流 ----------------
try:
    with open(WORKFLOW_FILE, "r", encoding="utf-8") as f:
        base_workflow = json.load(f)
    logger.info(f"成功加载工作流文件: {WORKFLOW_FILE}")
except FileNotFoundError:
    logger.error(f"工作流文件不存在: {WORKFLOW_FILE}")
    raise
except json.JSONDecodeError as e:
    logger.error(f"工作流文件JSON解析错误: {e}")
    raise

# ---------------- 请求体定义 ----------------
class GenerateVideoRequest(BaseModel):
    positive: str
    negative: str = ""
    height: int = 512
    width: int = 512
    length: int = 81  # 视频帧数
    fps: int = 16

# ---------------- FastAPI 实例 ----------------
app = FastAPI(title="ComfyUI Video Generation API")

# ---------------- 辅助函数 ----------------
def wait_for_prompt_completion(prompt_id: str, max_wait: int = MAX_WAIT_TIME) -> dict:
    """
    等待 ComfyUI 任务完成
    """
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            history_resp = requests.get(f"{COMFYUI_API_URL}/history/{prompt_id}")
            history_resp.raise_for_status()
            history = history_resp.json()
            
            if prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("status_str") == "success":
                    logger.info(f"任务 {prompt_id} 已完成")
                    return history[prompt_id]
                elif status.get("status_str") == "error":
                    logger.error(f"任务 {prompt_id} 执行失败: {status}")
                    raise HTTPException(status_code=500, detail=f"ComfyUI task failed: {status}")
            
            time.sleep(POLL_INTERVAL)
        except requests.RequestException as e:
            logger.error(f"查询任务状态时出错: {e}")
            raise HTTPException(status_code=500, detail=f"Error checking task status: {e}")
    
    logger.error(f"任务 {prompt_id} 超时未完成")
    raise HTTPException(status_code=504, detail="Task timeout")

def download_file_from_comfyui(file_info: dict) -> str:
    """
    从 ComfyUI 下载文件并保存到本地
    """
    try:
        filename = file_info["filename"]
        subfolder = file_info.get("subfolder", "")
        filetype = file_info.get("type", "output")
        
        # 构建下载URL
        file_url = f"{COMFYUI_API_URL}/view?filename={urllib.parse.quote(filename)}"
        if subfolder:
            file_url += f"&subfolder={urllib.parse.quote(subfolder)}"
        if filetype:
            file_url += f"&type={urllib.parse.quote(filetype)}"
        
        logger.info(f"下载文件: {filename} (子文件夹: {subfolder})")
        
        # 下载文件
        file_resp = requests.get(file_url)
        file_resp.raise_for_status()
        
        # 确保子目录存在
        save_dir = os.path.join(STATIC_DIR, subfolder) if subfolder else STATIC_DIR
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存文件
        save_path = os.path.join(save_dir, filename)
        with open(save_path, "wb") as f:
            f.write(file_resp.content)
        
        file_size = len(file_resp.content) / 1024 / 1024
        logger.info(f"文件已保存: {save_path} (大小: {file_size:.2f} MB)")
        
        # 生成对外访问URL
        public_url = f"{SERVER_BASE_URL}/static/{subfolder}/{filename}" if subfolder else f"{SERVER_BASE_URL}/static/{filename}"
        return public_url
        
    except Exception as e:
        logger.error(f"文件下载失败: {e}")
        raise HTTPException(status_code=500, detail=f"File download failed: {e}")

# ---------------- 生成接口 ----------------
@app.post("/generate_video", response_model=dict)
def generate_video(req: GenerateVideoRequest, x_api_key: str = Header(None)):
    """
    生成视频的API端点
    """
    # API密钥验证
    if x_api_key != API_KEY:
        logger.warning(f"无效的API密钥: {x_api_key}")
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    logger.info(f"收到视频生成请求: {req.positive[:50]}...")
    
    # 深拷贝工作流
    workflow = copy.deepcopy(base_workflow)
    
    # 修改工作流参数
    workflow["16"]["inputs"]["positive_prompt"] = req.positive
    workflow["16"]["inputs"]["negative_prompt"] = req.negative 
    workflow["37"]["inputs"]["width"] = req.width
    workflow["37"]["inputs"]["height"] = req.height
    workflow["37"]["inputs"]["num_frames"] = req.length
    workflow["58"]["inputs"]["frame_rate"] = req.fps
    
    logger.info(f"工作流配置 - 尺寸: {req.width}x{req.height}, 帧数: {req.length}, FPS: {req.fps}")
    
    # 提交任务到ComfyUI
    try:
        prompt_resp = requests.post(
            f"{COMFYUI_API_URL}/prompt",
            json={"prompt": workflow}
        )
        prompt_resp.raise_for_status()
        prompt_id = prompt_resp.json()["prompt_id"]
        logger.info(f"任务已提交，Prompt ID: {prompt_id}")
    except requests.RequestException as e:
        logger.error(f"提交任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit task: {e}")
    
    # 等待任务完成
    task_result = wait_for_prompt_completion(prompt_id)
    
    # 提取输出文件
    outputs = task_result.get("outputs", {})
    if not outputs:
        logger.error("任务完成但未找到输出")
        raise HTTPException(status_code=500, detail="No outputs found")
    
    # 下载所有输出文件（视频和图片）
    public_urls = []
    for node_id, output_data in outputs.items():
        # # 检查图片输出
        # if "images" in output_data:
        #     for file_info in output_data["images"]:
        #         try:
        #             url = download_file_from_comfyui(file_info)
        #             public_urls.append({
        #                 "node_id": node_id,
        #                 "type": "image",
        #                 "url": url,
        #                 "filename": file_info["filename"]
        #             })
        #         except Exception as e:
        #             logger.warning(f"下载图片失败 (节点 {node_id}): {e}")
        
        # # 检查视频输出
        # if "video" in output_data:
        #     for file_info in output_data["video"]:
        #         try:
        #             url = download_file_from_comfyui(file_info)
        #             public_urls.append({
        #                 "node_id": node_id,
        #                 "type": "video",
        #                 "url": url,
        #                 "filename": file_info["filename"]
        #             })
        #         except Exception as e:
        #             logger.warning(f"下载视频失败 (节点 {node_id}): {e}")

        # 已保存视频，直接复制过来
        comfyui_save_video_path = output_data['gifs'][0]['fullpath']
        save_video_path = os.path.join(STATIC_DIR, output_data['gifs'][0]['filename'])
        shutil.copy(comfyui_save_video_path, os.path.join(STATIC_DIR, output_data['gifs'][0]['filename']))
        public_urls.append(
            {
                "node_id": node_id,
                "type": "video",
                "url": f"{SERVER_BASE_URL}/static/{output_data['gifs'][0]['filename']}",
                "filename": save_video_path
            }
        )
    
    if not public_urls:
        logger.error("未找到任何可下载的文件")
        raise HTTPException(status_code=500, detail="No downloadable files found")
    
    logger.info(f"任务完成，生成 {len(public_urls)} 个文件")
    return {
        "prompt_id": prompt_id,
        "status": "success",
        "files": public_urls
    }

# ---------------- 静态文件服务 ----------------
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------------- 健康检查 ----------------
@app.get("/health")
def health_check():
    """
    健康检查端点
    """
    try:
        # 检查ComfyUI是否可访问
        resp = requests.get(f"{COMFYUI_API_URL}/system_stats", timeout=5)
        resp.raise_for_status()
        return {
            "status": "healthy",
            "comfyui_status": "connected",
            "static_dir": STATIC_DIR,
            "server_url": SERVER_BASE_URL
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=503, detail="ComfyUI not available")

# ---------------- 启动信息 ----------------
if __name__ == "__main__":
    import uvicorn
    logger.info(f"启动API服务器，监听端口 9001")
    logger.info(f"ComfyUI地址: {COMFYUI_API_URL}")
    logger.info(f"工作流文件: {WORKFLOW_FILE}")
    logger.info(f"视频存储目录: {STATIC_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=9003)