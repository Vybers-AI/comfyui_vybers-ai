# Quick Start

```
# activate env
source /workspace/miniconda3/bin/activate comfyui
# Terminal 1, run comfyui main process
python main.py --listen --use-sage-attention --port 8188

# Terminal 2, run wan2.1 1.3B t2v server
# workflow path set in WORKFLOW_FILE in demo_comfyui_fastapi_wan21_1__3B.py
uvicorn demo_comfyui_fastapi_wan21_1__3B:app --host 0.0.0.0 --port 9003 --reload --log-level debug

# api demo
python demo_api.py
```

# For develop workflow, enable Manager
```
python main.py --enable-manager --use-sage-attention
```


