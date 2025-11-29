import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load your API key
load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

print("Checking available models...")
try:
    # List all models available to your API key
    for m in genai.list_models():
        # Only show models that can generate text (chat)
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error: {e}")