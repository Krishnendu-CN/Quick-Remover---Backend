# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from rembg import remove, new_session
from sqlalchemy.orm import Session
from typing import Optional
import re

# Import from auth module
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

# Create rembg session once
session = new_session("isnet-general-use")

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

@app.post("/remove-bg")
async def remove_bg(
    image: UploadFile = File(...),
    # current_user: User = Depends(get_current_user)
):
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type")

    data = await image.read()

    if len(data) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        input_img = Image.open(BytesIO(data))
        output_img = remove(input_img, session=session)
        output_img = output_img.convert("RGBA")

        out_io = BytesIO()
        output_img.save(out_io, format="PNG")
        out_io.seek(0)
        return StreamingResponse(out_io, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Utility function
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

@app.post("/save-edited")
async def save_edited(
    image: UploadFile = File(...),
    brightness: float = Form(100),
    contrast: float = Form(100),
    saturation: float = Form(100),
    hue: float = Form(0),
    rotation: float = Form(0),
    flip_horizontal: bool = Form(False),
    flip_vertical: bool = Form(False),
    blur: float = Form(0),
    sepia: float = Form(0),
    bg_type: str = Form("transparent"),
    bg_color: str = Form(None),
    # current_user: User = Depends(get_current_user)
):
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type")

    data = await image.read()
    if len(data) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        # Open image and ensure RGBA mode for transparency
        img = Image.open(BytesIO(data)).convert("RGBA")
        
        # Store alpha channel for later use
        alpha_channel = img.getchannel('A') if img.mode == 'RGBA' else None
        
        # Apply transformations
        if rotation != 0:
            img = img.rotate(rotation, expand=True)
            if alpha_channel:
                alpha_channel = alpha_channel.rotate(rotation, expand=True)
        
        if flip_horizontal:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if alpha_channel:
                alpha_channel = alpha_channel.transpose(Image.FLIP_LEFT_RIGHT)
        
        if flip_vertical:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if alpha_channel:
                alpha_channel = alpha_channel.transpose(Image.FLIP_TOP_BOTTOM)
        
        # For filters, we need to handle transparency properly
        if brightness != 100 or contrast != 100 or saturation != 100 or hue != 0:
            # Convert to RGB for filter operations, then back to RGBA
            rgb_img = img.convert('RGB')
            
            if brightness != 100:
                enhancer = ImageEnhance.Brightness(rgb_img)
                rgb_img = enhancer.enhance(brightness / 100)
            
            if contrast != 100:
                enhancer = ImageEnhance.Contrast(rgb_img)
                rgb_img = enhancer.enhance(contrast / 100)
            
            if saturation != 100:
                enhancer = ImageEnhance.Color(rgb_img)
                rgb_img = enhancer.enhance(saturation / 100)
            
            if hue != 0:
                # Convert to HSV, adjust hue, then convert back to RGB
                hsv_img = rgb_img.convert("HSV")
                h, s, v = hsv_img.split()
                h = h.point(lambda x: (x + int(hue * 255 / 360)) % 255)
                hsv_img = Image.merge("HSV", (h, s, v))
                rgb_img = hsv_img.convert("RGB")
            
            # Convert back to RGBA and restore alpha channel
            img = rgb_img.convert('RGBA')
            if alpha_channel:
                # Ensure alpha channel matches the size of the processed image
                if alpha_channel.size == img.size:
                    r, g, b, _ = img.split()
                    img = Image.merge('RGBA', (r, g, b, alpha_channel))
        
        if blur > 0:
            # Apply blur to the RGB channels only, preserve alpha
            if alpha_channel:
                r, g, b, a = img.split()
                rgb_img = Image.merge('RGB', (r, g, b))
                blurred_rgb = rgb_img.filter(ImageFilter.GaussianBlur(radius=blur))
                r_blur, g_blur, b_blur = blurred_rgb.split()
                img = Image.merge('RGBA', (r_blur, g_blur, b_blur, a))
            else:
                img = img.filter(ImageFilter.GaussianBlur(radius=blur))
        
        if sepia > 0:
            sepia_intensity = sepia / 100
            width, height = img.size
            
            # Create a copy to work with
            sepia_img = img.copy()
            pixels = sepia_img.load()
            
            for py in range(height):
                for px in range(width):
                    r, g, b, a = pixels[px, py]
                    
                    # Only apply sepia to non-transparent pixels
                    if a > 0:
                        tr = min(255, int((r * (1 - (0.607 * sepia_intensity))) + 
                                         (g * (0.769 * sepia_intensity)) + 
                                         (b * (0.189 * sepia_intensity))))
                        tg = min(255, int((r * (0.349 * sepia_intensity)) + 
                                         (g * (1 - (0.314 * sepia_intensity))) + 
                                         (b * (0.168 * sepia_intensity))))
                        tb = min(255, int((r * (0.272 * sepia_intensity)) + 
                                         (g * (0.534 * sepia_intensity)) + 
                                         (b * (1 - (0.869 * sepia_intensity)))))
                        pixels[px, py] = (tr, tg, tb, a)
            
            img = sepia_img

        if bg_type == "color" and bg_color:
            bg_rgb = hex_to_rgb(bg_color)
            bg_img = Image.new("RGBA", img.size, bg_rgb + (255,))
            
            # Composite the foreground image over the background
            # This handles transparency properly
            img = Image.alpha_composite(bg_img, img)

        # Ensure we're saving as PNG with transparency
        out_io = BytesIO()
        img.save(out_io, format="PNG", optimize=True)
        out_io.seek(0)
        
        return StreamingResponse(out_io, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/signup")
def signup(username: str = Form(...), email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    # Check if user already exists
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already taken")

    # Create new user
    new_user = User(
        username=username,
        email=email,
        password_hash=hash_password(password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Create access token
    token = create_access_token({"sub": email})
    
    return {
        "message": "User created successfully",
        "username": new_user.username,
        "access_token": token,
        "token_type": "bearer"
    }

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # Try to find user by email first, then by username
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user:
        user = db.query(User).filter(User.username == form_data.username).first()
    
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email/username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

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