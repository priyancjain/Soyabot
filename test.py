import os

HF_TOKEN = os.getenv("HF_TOKEN")  # Access environment variable
if HF_TOKEN:
    print("Token found!")
else:
    print("Token not found!")