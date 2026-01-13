import requests
import base64
from PIL import Image
from io import BytesIO
import time

headers = {"x-api-key": "12345"}


# 文生视频
# 先启动服务 
# uvicorn demo_comfyui_fastapi_wan21_1__3B:app --host 0.0.0.0 --port 9003 --reload --log-level debug
t1 = time.time()
resp = requests.post(
    "http://0.0.0.0:9003/generate_video",
    headers=headers,
    json={
        "positive": "Leopards hunt on the grassland",
        "negative": "",
        "height": 480,
        "width": 640,
        "length": 81,
        "fps": 16,
    }
)
print('infer time:',time.time()-t1)
result_video_path = resp.json()["files"][0]['filename']
print(result_video_path)