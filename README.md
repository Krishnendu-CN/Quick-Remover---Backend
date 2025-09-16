# 🚀 Quick BG Remover

A simple **background removal application** built with **FastAPI (Python backend)** and **React (frontend)**.

---

## 📦 Features
- Upload an image and remove background instantly
- FastAPI backend with Uvicorn server
- React frontend for user-friendly interface
- Supports image formats: JPG, PNG, JPEG
- Extensible for authentication & database (SQLAlchemy)

---

## ⚙️ Installation Guide

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/quick-bg-remover.git
cd quick-bg-remover/bg-remove-backend
```

---

### 2️⃣ Install Python & pip
Make sure you have **Python 3.10+** installed:
```bash
python3 --version
```

If not installed:
```bash
sudo apt update
sudo apt install python3 python3-pip -y
```

---

### 3️⃣ Install Dependencies
```bash
pip install --upgrade pip
pip install fastapi uvicorn[standard] rembg pillow python-multipart \
    sqlalchemy passlib[bcrypt] python-jose
```

👉 If you face issues with SQLAlchemy, upgrade it:
```bash
pip install --upgrade "sqlalchemy>=2.0"
```

---

### 4️⃣ Run the Backend
Navigate to backend folder and run:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- `--reload` → Auto restart on code change
- `--host 0.0.0.0` → Allow external access
- `--port 8000` → Run on port 8000

---

### 5️⃣ Access API Docs
Open your browser:
```
http://127.0.0.1:8000/docs
```