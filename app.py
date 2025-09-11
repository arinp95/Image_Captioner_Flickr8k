from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import os
import uuid
import aiofiles
import logging

from generate_captions import CaptionGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI(title="Image Captioning API")

# Allow CORS if deploying behind reverse proxy or for frontend apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory setup
upload_dir = "static/uploads"
os.makedirs(upload_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global CaptionGenerator instance
caption_generator: CaptionGenerator = None

@app.on_event("startup")
def load_models():
    global caption_generator
    model_path = "model.keras"
    tokenizer_path = "tokenizer.pkl"
    config_path = "config.pkl"

    caption_generator = CaptionGenerator(model_path, tokenizer_path, config_path)
    logger.info("Models loaded successfully")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

from fastapi import HTTPException

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, image: UploadFile = File(...)):
    if not image.filename or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    filename = f"{uuid.uuid4()}_{image.filename}"
    filepath = os.path.join(upload_dir, filename)

    # Read content once to check size and save
    content = await image.read()
    if len(content) > 5 * 1024 * 1024:  # 5MB limit
        raise HTTPException(status_code=400, detail="File too large")

    async with aiofiles.open(filepath, 'wb') as out_file:
        await out_file.write(content)

    try:
        photo = caption_generator.extract_features(filepath)
        caption = caption_generator.generate_caption(photo)
    except Exception as e:
        logger.error(f"Error during caption generation: {e}")
        raise HTTPException(status_code=500, detail="Error generating caption")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "caption": caption,
        "image_path": f"/static/uploads/{filename}"
    })

