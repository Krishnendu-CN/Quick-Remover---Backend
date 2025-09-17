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
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "https://quick-remover-backend.onrender.com" , "https://quick-remover-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_SIZE = 600 * 1024  # 200KB
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
        # Load as OpenCV BGR
        file_bytes = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if bgr is None:
            raise HTTPException(status_code=400, detail="Could not read image")

        h, w = bgr.shape[:2]

        # Init mask for GrabCut
        mask = np.zeros((h, w), np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        rect = (10, 10, w - 20, h - 20)
        cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        # Foreground mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

        # Create alpha channel
        alpha = (mask2 * 255).astype(np.uint8)

        # Merge BGR + Alpha
        b, g, r = cv2.split(bgr)
        bgra = cv2.merge([b, g, r, alpha])

        # Convert BGR(A) -> RGB(A) to fix "blue skin" issue
        rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)

        # Smooth alpha edges
        alpha_blurred = cv2.GaussianBlur(rgba[:, :, 3], (5, 5), 0)
        rgba[:, :, 3] = alpha_blurred

        # Save to BytesIO
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
        # Always open as RGBA
        img = Image.open(BytesIO(data)).convert("RGBA")

        # Keep alpha separate so it doesn’t get lost
        alpha_channel = img.getchannel("A")

        # ---------- Rotation ----------
        if rotation != 0:
            img = img.rotate(rotation, expand=True)
            alpha_channel = alpha_channel.rotate(rotation, expand=True)

        # ---------- Flip ----------
        if flip_horizontal:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            alpha_channel = alpha_channel.transpose(Image.FLIP_LEFT_RIGHT)
        if flip_vertical:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            alpha_channel = alpha_channel.transpose(Image.FLIP_TOP_BOTTOM)

        # ---------- Brightness / Contrast / Saturation ----------
        if brightness != 100 or contrast != 100 or saturation != 100:
            rgb_img = img.convert("RGB")
            if brightness != 100:
                rgb_img = ImageEnhance.Brightness(rgb_img).enhance(brightness / 100)
            if contrast != 100:
                rgb_img = ImageEnhance.Contrast(rgb_img).enhance(contrast / 100)
            if saturation != 100:
                rgb_img = ImageEnhance.Color(rgb_img).enhance(saturation / 100)
            # Reattach alpha
            img = rgb_img.convert("RGBA")
            img.putalpha(alpha_channel)

        # ---------- Blur ----------
        if blur > 0:
            r, g, b, a = img.split()
            blurred_rgb = Image.merge("RGB", (r, g, b)).filter(
                ImageFilter.GaussianBlur(radius=blur)
            )
            r_blur, g_blur, b_blur = blurred_rgb.split()
            img = Image.merge("RGBA", (r_blur, g_blur, b_blur, a))

        # ---------- Sepia ----------
        if sepia > 0:
            sepia_intensity = sepia / 100
            width, height = img.size
            sepia_img = img.copy()
            pixels = sepia_img.load()
            for y in range(height):
                for x in range(width):
                    r, g, b, a = pixels[x, y]
                    if a > 0:
                        tr = min(
                            255,
                            int(
                                (r * (1 - 0.607 * sepia_intensity))
                                + (g * 0.769 * sepia_intensity)
                                + (b * 0.189 * sepia_intensity)
                            ),
                        )
                        tg = min(
                            255,
                            int(
                                (r * 0.349 * sepia_intensity)
                                + (g * (1 - 0.314 * sepia_intensity))
                                + (b * 0.168 * sepia_intensity)
                            ),
                        )
                        tb = min(
                            255,
                            int(
                                (r * 0.272 * sepia_intensity)
                                + (g * 0.534 * sepia_intensity)
                                + (b * (1 - 0.869 * sepia_intensity))
                            ),
                        )
                        pixels[x, y] = (tr, tg, tb, a)
            img = sepia_img

        # ---------- Background ----------
        if bg_type == "color" and bg_color:
            bg_rgb = hex_to_rgb(bg_color)
            bg_img = Image.new("RGBA", img.size, bg_rgb + (255,))
            img = Image.alpha_composite(bg_img, img)

        # ---------- Output ----------
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
