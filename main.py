import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Ensure this import path exists in your project structure
from backend.routes.chat_routes import router as chat_router

app = FastAPI()

# --- FIX START: Define Base Directory ---
# This gets the folder where main.py is located (C:\Users\HP\...\ChatBot)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# --- FIX END ---

# Set up the limiter with 500 requests per minute
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend index.html file
@app.get("/")
def home():
    # --- FIX: Build the correct path using the OS separator ---
    file_path = os.path.join(BASE_DIR, "frontend", "index.html")
    return FileResponse(file_path)

# Mount static files
# --- FIX: Use absolute path for safety ---
static_path = os.path.join(BASE_DIR, "frontend")
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Include chat router
app.include_router(chat_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)