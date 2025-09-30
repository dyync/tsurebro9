import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pipeline import image_to_3d, get_seed
import os
import logging
from datetime import datetime

app = FastAPI()
logging.basicConfig(level=logging.INFO)

@app.get("/")
async def root():
    return {"message": "TRELLIS backend is running"}

@app.post("/generate")
async def generate(request: Request):
    req_data = await request.json()
    logging.info(f"[{datetime.now()}] Received request: {req_data}")
    seed = get_seed(req_data["randomize_seed"], req_data["seed"])
    result = image_to_3d(**req_data, seed=seed)
    return JSONResponse(content={"status": "success", "result": result})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
