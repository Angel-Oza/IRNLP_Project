"""
Pytest global configuration and environment setup.
"""

import os

# Prevent OpenMP and PyTorch thread collision on macOS ARM64
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
