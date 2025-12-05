# Setup Guide

Complete setup instructions for the Teaching Small Models to Reason project.

## Prerequisites

- **Python 3.8+** (Python 3.10+ recommended)
- **Ollama** installed and running
- **CUDA-capable GPU** (recommended for training) or CPU (much slower)
- **At least 8GB GPU memory** (for training with 4-bit quantization)
- **Internet connection** (for downloading models and dependencies)

## Step 1: Install Ollama

Visit [https://ollama.ai/](https://ollama.ai/) and follow installation instructions for your OS.

```bash
# Verify installation
ollama --version

# Start Ollama (if not running)
ollama serve
```

## Step 2: Pull a Model

Choose a model based on your needs:

```bash
# Option 1: Phi-3 (smaller, faster, recommended for testing)
ollama pull phi3

# Option 2: Llama 3 8B (better quality, slower)
ollama pull llama3:8b

# Option 3: Other models
ollama pull mistral
ollama pull codellama
```

**Recommended**: Start with `phi3` for faster iteration.

## Step 3: Set Up Python Environment

### Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install manually:

```bash
pip install z3-solver transformers peft datasets torch accelerate bitsandbytes
```

## Step 4: Verify Installation

Run these checks to ensure everything is set up correctly:

```bash
# Check Z3
python -c "import z3; print('Z3 version:', z3.get_version_string())"

# Check transformers
python -c "import transformers; print('Transformers version:', transformers.__version__)"

# Check Ollama connection
curl http://localhost:11434/api/tags

# Or test with Python
python -c "import requests; r = requests.get('http://localhost:11434/api/tags'); print('Ollama:', 'connected' if r.status_code == 200 else 'not connected')"
```

## Step 5: Quick Test

Test the pipeline with a small dataset:

```bash
# Generate a small test dataset
python scripts/dataset_generation/generate_examples.py

# This will create files in data/raw/ (may take a few minutes)
```

## Common Issues and Solutions

### Ollama Not Connecting

**Problem**: Scripts can't connect to Ollama.

**Solutions**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve

# Check if port 11434 is available
lsof -i :11434  # On macOS/Linux
netstat -ano | findstr :11434  # On Windows
```

### CUDA Out of Memory

**Problem**: GPU runs out of memory during training.

**Solutions**:
- Reduce batch size: Use `--batch-size 2` or `--batch-size 1`
- Ensure 4-bit quantization is enabled (default in finetune.py)
- Reduce max sequence length: `--max-length 1024`
- Use CPU (much slower): Ensure no GPU is used

### Model Download Issues

**Problem**: Can't download models via Ollama.

**Solutions**:
1. Check internet connection
2. Try a different model (phi3 is usually smaller/faster to download)
3. Manually download from HuggingFace and use the path
4. Check Ollama logs: `ollama logs`

### Import Errors

**Problem**: Python import errors when running scripts.

**Solutions**:
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check Python version
python --version  # Should be 3.8+

# Recreate virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Z3 Validation Errors

**Problem**: Z3 validation script fails.

**Solutions**:
```bash
# Reinstall Z3
pip uninstall z3-solver
pip install z3-solver

# Test Z3 directly
python -c "from z3 import Solver, parse_smt2_string; print('Z3 works')"
```

### Permission Errors

**Problem**: Permission denied when writing files.

**Solutions**:
```bash
# Check directory permissions
ls -la data/

# Fix permissions if needed
chmod -R 755 data/
chmod -R 755 scripts/
```

## Environment Variables (Optional)

Set these for convenience:

```bash
# Ollama configuration
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="phi3"  # or your preferred model

# Add to ~/.bashrc or ~/.zshrc for persistence
```

## System Requirements

### Minimum Requirements (CPU only)
- **CPU**: 4+ cores
- **RAM**: 16GB
- **Storage**: 20GB free space
- **Training Time**: ~6-12 hours for small dataset

### Recommended (GPU)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060, RTX 4060, etc.)
- **CUDA**: 11.8+ or 12.0+
- **RAM**: 16GB+
- **Storage**: 50GB free space
- **Training Time**: ~1-2 hours for small dataset

## Next Steps

After setup is complete:

1. **Run full pipeline**:
   ```bash
   python scripts/run_pipeline.py
   ```

2. **Read implementation details**:
   See [IMPLEMENTATION.md](IMPLEMENTATION.md)

3. **Check file structure**:
   See [ESSENTIAL_FILES.md](ESSENTIAL_FILES.md)

## Getting Help

If you encounter issues:

1. Check this SETUP.md for common solutions
2. Review error messages carefully
3. Check logs in the terminal output
4. Verify all prerequisites are installed
5. Try running individual scripts to isolate the issue

## Verification Checklist

Before running the full pipeline, verify:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Ollama installed and running
- [ ] At least one model pulled (`ollama pull phi3`)
- [ ] Z3 installed and working
- [ ] GPU available (if using GPU) or CPU acceptable (if using CPU)
- [ ] Sufficient disk space (20GB+)
- [ ] All verification tests pass

Once all checks pass, you're ready to run the pipeline!
