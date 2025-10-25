import os
from dotenv import load_dotenv

load_dotenv()

# Models
SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "microsoft/Phi-3-mini-128k-instruct")

# Quantization
USE_4BIT = os.getenv("USE_4BIT", True)
