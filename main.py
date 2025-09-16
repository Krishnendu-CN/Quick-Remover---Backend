# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
from sqlalchemy.orm import Session
import cv2
import numpy as np

# Import from your auth module
from auth import SessionLocal, User, hash_password, verify_password, create_access_token, verify_token

app = FastAPI(title="BG Remove API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_SIZE = 12 * 1024 * 1024  # 12MB
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Dependency to get current user
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = verify_token(token)
    if payload is None:
        raise credentials_exception
    
    email: str = payload.get("sub")
    if email is None:
        raise credentials_exception
    
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    
    return user

# Utility function to convert hex color to RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# -----------------------------
# Background removal endpoint
# -----------------------------
@app.post("/remove-bg")
async def remove_bg(image: UploadFile = File(...)):
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type")

    data = await image.read()
    if len(data) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        pil_img = Image.open(BytesIO(data)).convert("RGBA")
        img = np.array(pil_img)

        # Convert RGBA to BGR for OpenCV
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        # Convert to HSV for background masking
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

        # Example: remove white background
        lower = np.array([0, 0, 200])
        upper = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower, upper)
        mask_inv = cv2.bitwise_not(mask)

        # Create alpha channel
        alpha = mask_inv.astype(np.uint8)

        # Merge channels
        b, g, r = cv2.split(bgr_img)
        rgba = cv2.merge([r, g, b, alpha])

        out_img = Image.fromarray(rgba)

        out_io = BytesIO()
        out_img.save(out_io, format="PNG")
        out_io.seek(0)
        return StreamingResponse(out_io, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Save edited image endpoint
# -----------------------------
@app.post("/save-edited")
async def save_edited(
    image: UploadFile = File(...),
    brightness: float = Form(100),
    contrast: float = Form(100),
    saturation: float = Form(100),
    rotation: float = Form(0),
    flip_horizontal: bool = Form(False),
    flip_vertical: bool = Form(False),
    blur: float = Form(0),
    sepia: float = Form(0),
    bg_type: str = Form("transparent"),
    bg_color: str = Form(None),
):
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type")

    data = await image.read()
    if len(data) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        img = Image.open(BytesIO(data)).convert("RGBA")
        alpha_channel = img.getchannel('A') if img.mode == 'RGBA' else None

        # Rotation
        if rotation != 0:
            img = img.rotate(rotation, expand=True)
            if alpha_channel:
                alpha_channel = alpha_channel.rotate(rotation, expand=True)

        # Flip
        if flip_horizontal:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if alpha_channel:
                alpha_channel = alpha_channel.transpose(Image.FLIP_LEFT_RIGHT)
        if flip_vertical:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if alpha_channel:
                alpha_channel = alpha_channel.transpose(Image.FLIP_TOP_BOTTOM)

        # Brightness, Contrast, Saturation
        if brightness != 100 or contrast != 100 or saturation != 100:
            rgb_img = img.convert('RGB')
            if brightness != 100:
                rgb_img = ImageEnhance.Brightness(rgb_img).enhance(brightness/100)
            if contrast != 100:
                rgb_img = ImageEnhance.Contrast(rgb_img).enhance(contrast/100)
            if saturation != 100:
                rgb_img = ImageEnhance.Color(rgb_img).enhance(saturation/100)
            img = rgb_img.convert('RGBA')
            if alpha_channel and alpha_channel.size == img.size:
                r, g, b, _ = img.split()
                img = Image.merge('RGBA', (r, g, b, alpha_channel))

        # Blur
        if blur > 0:
            if alpha_channel:
                r, g, b, a = img.split()
                blurred_rgb = Image.merge('RGB', (r, g, b)).filter(ImageFilter.GaussianBlur(radius=blur))
                r_blur, g_blur, b_blur = blurred_rgb.split()
                img = Image.merge('RGBA', (r_blur, g_blur, b_blur, a))
            else:
                img = img.filter(ImageFilter.GaussianBlur(radius=blur))

        # Sepia
        if sepia > 0:
            sepia_intensity = sepia / 100
            width, height = img.size
            sepia_img = img.copy()
            pixels = sepia_img.load()
            for y in range(height):
                for x in range(width):
                    r, g, b, a = pixels[x, y]
                    if a > 0:
                        tr = min(255, int((r*(1-0.607*sepia_intensity)) + (g*0.769*sepia_intensity) + (b*0.189*sepia_intensity)))
                        tg = min(255, int((r*0.349*sepia_intensity) + (g*(1-0.314*sepia_intensity)) + (b*0.168*sepia_intensity)))
                        tb = min(255, int((r*0.272*sepia_intensity) + (g*0.534*sepia_intensity) + (b*(1-0.869*sepia_intensity))))
                        pixels[x, y] = (tr, tg, tb, a)
            img = sepia_img

        # Background color
        if bg_type == "color" and bg_color:
            bg_rgb = hex_to_rgb(bg_color)
            bg_img = Image.new("RGBA", img.size, bg_rgb + (255,))
            img = Image.alpha_composite(bg_img, img)

        out_io = BytesIO()
        img.save(out_io, format="PNG", optimize=True)
        out_io.seek(0)
        return StreamingResponse(out_io, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True}


# -----------------------------
# Auth endpoints
# -----------------------------
@app.post("/signup")
def signup(username: str = Form(...), email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already taken")

    new_user = User(
        username=username,
        email=email,
        password_hash=hash_password(password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_access_token({"sub": email})
    return {
        "message": "User created successfully",
        "username": new_user.username,
        "access_token": token,
        "token_type": "bearer"
    }


@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user:
        user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email/username or password", headers={"WWW-Authenticate": "Bearer"})
    token = create_access_token({"sub": user.email})
    return {"access_token": token, "token_type": "bearer"}


@app.get("/users/me")
def read_users_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
