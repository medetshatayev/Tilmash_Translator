#!/bin/bash
# Startup script for Translator app with safe multiuser configuration

  # Get defaults from Python config module
  echo "Loading defaults from config.py..."
  eval $(python3 -c "from config import ENV_DEFAULTS; print(';'.join([f'export {k}={v}' for k,v in ENV_DEFAULTS.items()]))")

# Create model directories
mkdir -p local_llms/instances
mkdir -p local_llms/locks

# Clean up ALL instance files at startup
rm -f local_llms/instances/*.gguf

# Start the Streamlit app with proper configuration
# Add memory limits to prevent OOM crashes
export MALLOC_ARENA_MAX=4  # Limit memory fragmentation in glibc malloc
export PYTHONMALLOC=malloc  # Use system malloc which is more stable

# Suppress Metal/GPU initialization noise but keep model load/unload logs
export GGML_METAL_VERBOSE=0
export GGML_METAL_DEBUG=0

# Start Streamlit with appropriate settings - ensure proper quoting
streamlit run main.py --server.port=8501 --server.fileWatcherType=none --server.address=0.0.0.0 2>&1 | grep -v -E "Metal|ggml|llama\.cpp"
