#!/usr/bin/env python3
"""
Example client for interacting with the vLLM OpenAI-compatible inference server.

Usage:
    python client_example.py <server_url> <api_key>
    
Environment Variables:
    VLLM_BASE_URL   - Server URL (e.g., https://your-endpoint)
    VLLM_API_KEY    - API key for authentication
    VLLM_MODEL_NAME - Model name (default: Qwen/Qwen3-0.6B)

Example:
    export VLLM_BASE_URL=https://your-workspace--vllm-qwen-inference-serve.modal.run
    export VLLM_API_KEY=sk-vllm-test-key-12345
    python client_example.py
"""

import sys
import os
import argparse
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(description="vLLM Client Example")
    parser.add_argument("server_url", nargs="?", help="Base URL of the vLLM server (e.g., https://your-endpoint)")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--model", help="Model name to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0-2)")
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum tokens to generate")
    parser.add_argument("--stream", action="store_true", help="Stream the response")
    
    args = parser.parse_args()
    
    # Get values from arguments or environment variables
    server_url = args.server_url or os.getenv("VLLM_BASE_URL")
    api_key = args.api_key or os.getenv("VLLM_API_KEY", "sk-vllm-test-key-12345")
    model = args.model or os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen3-0.6B")
    
    if not server_url:
        print("❌ Error: server_url is required!")
        print("\nProvide via:")
        print("  1. Argument: python client_example.py https://your-endpoint")
        print("  2. Environment: export VLLM_BASE_URL=https://your-endpoint")
        sys.exit(1)
    
    # Initialize the OpenAI client pointing to our vLLM server
    client = OpenAI(
        base_url=f"{server_url}/v1",
        api_key=api_key,
    )
    
    print(f"🔗 Connected to: {server_url}")
    print(f"🤖 Model: {model}")
    print(f"🔑 API Key: {api_key[:20]}...")
    print("-" * 60)
    
    # Get user prompt
    print("\nEnter your prompt (or type 'quit' to exit):")
    user_input = input("> ").strip()
    
    if user_input.lower() == "quit":
        print("Goodbye!")
        return
    
    print("\n📝 Generating response...\n")
    
    try:
        # Create chat completion
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stream=args.stream,
        )
        
        # Handle streaming vs non-streaming
        if args.stream:
            print("💬 Response (streaming):\n")
            for chunk in response:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print("\n")
        else:
            print("💬 Response:\n")
            print(response.choices[0].message.content)
            print(f"\n📊 Tokens used: {response.usage.total_tokens}")
        
        print("\n✅ Done!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def example_non_streaming():
    """Example: Non-streaming chat completion"""
    from dotenv import load_dotenv
    load_dotenv()
    
    server_url = os.getenv("VLLM_BASE_URL", "https://your-workspace--vllm-qwen-inference-serve.modal.run")
    api_key = os.getenv("VLLM_API_KEY", "sk-vllm-test-key-12345")
    
    client = OpenAI(
        base_url=f"{server_url}/v1",
        api_key=api_key,
    )
    
    response = client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=[
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a Python function to calculate factorial."}
        ],
        temperature=0.7,
        max_tokens=300,
    )
    
    print(response.choices[0].message.content)


def example_streaming():
    """Example: Streaming chat completion"""
    from dotenv import load_dotenv
    load_dotenv()
    
    server_url = os.getenv("VLLM_BASE_URL", "https://your-workspace--vllm-qwen-inference-serve.modal.run")
    api_key = os.getenv("VLLM_API_KEY", "sk-vllm-test-key-12345")
    
    client = OpenAI(
        base_url=f"{server_url}/v1",
        api_key=api_key,
    )
    
    stream = client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=[
            {"role": "user", "content": "Tell me a joke."}
        ],
        stream=True,
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


def example_text_completion():
    """Example: Text completion (non-chat)"""
    from dotenv import load_dotenv
    load_dotenv()
    
    server_url = os.getenv("VLLM_BASE_URL", "https://your-workspace--vllm-qwen-inference-serve.modal.run")
    api_key = os.getenv("VLLM_API_KEY", "sk-vllm-test-key-12345")
    
    client = OpenAI(
        base_url=f"{server_url}/v1",
        api_key=api_key,
    )
    
    response = client.completions.create(
        model="Qwen/Qwen3-0.6B",
        prompt="The capital of France is",
        max_tokens=10,
    )
    
    print(response.choices[0].text)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("📖 vLLM Client Example\n")
        print("Usage:")
        print("  python client_example.py <server_url> [--api-key KEY] [--model MODEL] [--stream]\n")
        print("Example:")
        print("  python client_example.py https://your-endpoint --api-key sk-vllm-test-key-12345\n")
        print("Or with environment variables:")
        print("  export VLLM_BASE_URL=https://your-endpoint")
        print("  export VLLM_API_KEY=sk-vllm-test-key-12345")
        print("  python client_example.py\n")
        print("View example functions in the source code:")
        print("  - example_non_streaming()")
        print("  - example_streaming()")
        print("  - example_text_completion()")
    else:
        main()
