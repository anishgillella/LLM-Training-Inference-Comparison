import os
import subprocess
import modal
from typing import Optional

# ============================================================================
# CONFIGURATION - FROM ENVIRONMENT VARIABLES WITH DEFAULTS
# ============================================================================

MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen3-0.6B")
MODEL_REVISION = os.getenv("VLLM_MODEL_REVISION", None)
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
N_GPU = int(os.getenv("VLLM_N_GPU", "1"))
FAST_BOOT = os.getenv("VLLM_FAST_BOOT", "true").lower() == "true"

# API Key for basic authentication (IMPORTANT: Set via environment variable in production!)
API_KEY = os.getenv("VLLM_API_KEY", "sk-vllm-test-key-12345")

# Timing constants
MINUTES = 60
SCALEDOWN_MINUTES = int(os.getenv("VLLM_SCALEDOWN_MINUTES", "10"))
TIMEOUT_MINUTES = int(os.getenv("VLLM_TIMEOUT_MINUTES", "10"))

# Log current configuration
print(f"""
╔════════════════════════════════════════════════════════════╗
║           vLLM Configuration (from environment)            ║
╚════════════════════════════════════════════════════════════╝
  Model:              {MODEL_NAME}
  Model Revision:     {MODEL_REVISION or "latest"}
  API Key:            {API_KEY[:20]}...
  GPU Count:          {N_GPU}
  FAST_BOOT:          {FAST_BOOT}
  Port:               {VLLM_PORT}
  Scaledown Minutes:  {SCALEDOWN_MINUTES}
  Timeout Minutes:    {TIMEOUT_MINUTES}
""")

# ============================================================================
# CONTAINER IMAGE SETUP
# ============================================================================

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.10.2",
        "huggingface_hub[hf_transfer]==0.35.0",
        "flashinfer-python==0.3.1",
        "torch==2.8.0",
        "uvicorn==0.30.0",
        "fastapi==0.115.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# ============================================================================
# VOLUMES FOR CACHING
# ============================================================================

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# ============================================================================
# MODAL APP DEFINITION
# ============================================================================

app = modal.App(name="vllm-qwen-inference")


# ============================================================================
# INFERENCE SERVER FUNCTION
# ============================================================================

@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=SCALEDOWN_MINUTES * MINUTES,
    timeout=TIMEOUT_MINUTES * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=16)
@modal.web_server(port=VLLM_PORT, startup_timeout=TIMEOUT_MINUTES * MINUTES)
def serve():
    """
    Start the vLLM OpenAI-compatible inference server.
    
    This function runs the vLLM server with the specified model and configuration.
    The server exposes OpenAI-compatible API endpoints.
    """
    
    # Build vLLM serve command
    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--max-model-len", "2048",  # Reasonable context for 0.6B model
    ]
    
    # Add optional model revision
    if MODEL_REVISION:
        cmd.extend(["--revision", MODEL_REVISION])
    
    # FAST_BOOT optimization: disable CUDA graphs and compilation
    if FAST_BOOT:
        cmd.append("--enforce-eager")
    else:
        cmd.append("--no-enforce-eager")
    
    # Tensor parallelism for multi-GPU (if N_GPU > 1)
    cmd.extend(["--tensor-parallel-size", str(N_GPU)])
    
    # Enable API key authentication via environment variable
    os.environ["VLLM_API_KEY"] = API_KEY
    
    print(f"🚀 Starting vLLM server with command:")
    print(f"   {' '.join(cmd)}")
    print(f"📊 Configuration:")
    print(f"   Model: {MODEL_NAME}")
    print(f"   GPU: H100 x {N_GPU}")
    print(f"   FAST_BOOT: {FAST_BOOT}")
    print(f"   Port: {VLLM_PORT}")
    print(f"   API Key: {API_KEY[:20]}...")
    
    # Run vLLM server
    subprocess.run(cmd, check=True)


# ============================================================================
# TEST FUNCTION
# ============================================================================

@app.function(
    image=modal.Image.debian_slim().pip_install("openai==1.76.0", "requests==2.32.3"),
    timeout=5 * MINUTES,
)
def test_inference(content: str = None):
    """
    Test the inference server with sample requests.
    
    Args:
        content: Optional custom content to test
    """
    import time
    import requests
    from openai import OpenAI
    
    # Wait for the server to start
    print("⏳ Waiting for server to start...")
    time.sleep(10)
    
    # Get the server URL from Modal
    server_url = f"http://localhost:{VLLM_PORT}"
    api_key = API_KEY
    
    print(f"\n✅ Testing inference server at {server_url}")
    print(f"🔑 Using API key: {api_key[:20]}...\n")
    
    # Test 1: Health check
    print("Test 1: Health Check")
    print("-" * 50)
    try:
        # vLLM doesn't have a /health endpoint by default, so we'll test with a model endpoint
        response = requests.get(f"{server_url}/v1/models", headers={"Authorization": f"Bearer {api_key}"})
        if response.status_code == 200:
            print("✓ Server is healthy")
            print(f"  Response: {response.json()}\n")
        else:
            print(f"✗ Server health check failed: {response.status_code}\n")
    except Exception as e:
        print(f"✗ Health check error: {e}\n")
    
    # Test 2: Chat completions
    print("Test 2: Chat Completions (Non-streaming)")
    print("-" * 50)
    try:
        client = OpenAI(
            base_url=f"{server_url}/v1",
            api_key=api_key,
        )
        
        test_content = content or "You are a helpful assistant. Explain quantum computing in 2 sentences."
        
        print(f"Prompt: {test_content}\n")
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": test_content}
            ],
            temperature=0.7,
            max_tokens=200,
        )
        
        print(f"Response:")
        print(f"  {response.choices[0].message.content}\n")
    except Exception as e:
        print(f"✗ Chat completion error: {e}\n")
    
    # Test 3: Streaming response
    print("Test 3: Chat Completions (Streaming)")
    print("-" * 50)
    try:
        client = OpenAI(
            base_url=f"{server_url}/v1",
            api_key=api_key,
        )
        
        print("Streaming response:\n")
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "Write a short haiku about AI."}
            ],
            temperature=0.7,
            max_tokens=100,
            stream=True,
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        
        print("\n\n✓ Streaming completed\n")
    except Exception as e:
        print(f"✗ Streaming error: {e}\n")
    
    print("=" * 50)
    print("✅ All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run test: modal run vllm_inference.py test
        content = None
        if len(sys.argv) > 2:
            content = " ".join(sys.argv[2:])
        test_inference.remote(content)
    else:
        print("""
        Usage:
        
        1. Deploy the server:
           modal deploy vllm_inference.py
        
        2. Run tests (local):
           modal run vllm_inference.py test
           modal run vllm_inference.py test "Your custom prompt here"
        
        Environment Variables:
        - VLLM_MODEL_NAME          Model to deploy (default: Qwen/Qwen3-0.6B)
        - VLLM_MODEL_REVISION      Model revision (default: latest)
        - VLLM_API_KEY             API key for auth (default: sk-vllm-test-key-12345)
        - VLLM_N_GPU               Number of GPUs (default: 1)
        - VLLM_FAST_BOOT           Enable fast startup (default: true)
        - VLLM_PORT                Server port (default: 8000)
        - VLLM_SCALEDOWN_MINUTES   Keep-alive minutes (default: 10)
        - VLLM_TIMEOUT_MINUTES     Startup timeout (default: 10)
        """)
