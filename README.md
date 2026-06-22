# NeuraScan — Brain Tumor MRI Classifier

A full-stack AI web application that classifies brain MRI scans into four categories using a fine-tuned Vision Transformer (ViT-Base). Built as a portfolio project demonstrating end-to-end ML engineering: model training, API development, containerization, and cloud deployment.

**Live Demo:** [neurascan-frontend.mangoforest-74886e44.southeastasia.azurecontainerapps.io](https://neurascan-frontend.mangoforest-74886e44.southeastasia.azurecontainerapps.io)

---

## What it does

Upload a brain MRI scan (JPEG or PNG) and the model predicts which of four conditions is present, along with a confidence score and full score breakdown across all classes:

- **Glioma** — tumor originating in the glial cells
- **Meningioma** — tumor in the membranes surrounding the brain
- **Pituitary** — tumor at the base of the brain
- **No Tumor** — healthy scan

---

## Architecture

```
React/Vite Frontend  →  FastAPI Backend  →  ViT-Base Model
(Azure Container Apps)   (Azure Container Apps)   (HuggingFace Hub)
```

The backend loads the fine-tuned model weights directly from HuggingFace Hub (`iddc/neurascan-vit`) at startup using `from_pretrained()`, ensuring the correct architecture and weights are always loaded as a matched unit.

---

## Model

| Property | Detail |
|---|---|
| Architecture | ViT-Base (google/vit-base-patch16-224-in21k) |
| Fine-tuned on | Brain Tumor MRI Dataset (Kaggle, 4 classes) |
| Test accuracy | ~94.3% |
| Training platform | Google Colab (T4 GPU) |
| Weights hosted | HuggingFace Hub (`iddc/neurascan-vit`) |

The backbone was initialized from Google's ImageNet-21k pretrained ViT-Base checkpoint. The classification head was replaced with a 4-class linear layer and fine-tuned end-to-end. Images are preprocessed with standard ImageNet normalization (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`) and resized to 224×224.

---

## Tech Stack

**Backend**
- Python 3.11, FastAPI, PyTorch (CPU), HuggingFace Transformers
- Containerized with Docker (CPU-only build, ~1.5GB image)
- Deployed to Azure Container Apps (Southeast Asia)

**Frontend**
- React 18, Vite, Tailwind CSS
- Containerized with Docker + nginx
- Deployed to Azure Container Apps (Southeast Asia)

**CI/CD**
- GitHub Actions — auto-builds and deploys backend on every push to `main`
- Docker images stored in Azure Container Registry (`neurascanacr2026`)

---

## API

Base URL: `https://neurascan-backend.mangoforest-74886e44.southeastasia.azurecontainerapps.io`

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server status and model info |
| `/classes` | GET | List of detectable conditions |
| `/predict` | POST | Upload MRI image, returns prediction |
| `/docs` | GET | Interactive API documentation (Swagger UI) |

Example response from `/predict`:

```json
{
  "predicted_class": "meningioma",
  "confidence": 0.983,
  "all_scores": {
    "glioma": 0.010,
    "meningioma": 0.983,
    "notumor": 0.002,
    "pituitary": 0.004
  },
  "low_confidence": false,
  "disclaimer": "Research tool only. Not a substitute for professional medical diagnosis."
}
```

---

## Local Development

**Prerequisites:** Python 3.11+, Node.js 18+, Git

**Backend**
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn api:app --reload --port 7860
```

**Frontend**
```bash
cd frontend
npm install
# Create .env with:
# VITE_API_URL=http://127.0.0.1:7860
npm run dev
```

**Docker (full stack)**
```bash
# Backend
docker build -t neurascan-backend ./backend
docker run -p 7860:7860 neurascan-backend

# Frontend
docker build -t neurascan-frontend ./frontend
docker run -p 5173:80 neurascan-frontend
```

---

## Project Structure

```
neurascan/
├── backend/
│   ├── api.py               # FastAPI app, inference pipeline
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main UI component
│   │   ├── main.jsx
│   │   └── index.css        # Tailwind entry
│   ├── package.json
│   ├── vite.config.js
│   └── Dockerfile
└── .github/
    └── workflows/           # GitHub Actions CI/CD
```

---

## Challenges & Notes

**Transformers version mismatch** — the model silently loaded with random weights after a library upgrade renamed ViT's internal attention modules. Every parameter key mismatched, `strict=False` swallowed the errors, and predictions were wrong with no visible indication. Fixed by downgrading to the version used during training and switching to `from_pretrained()` to load architecture and weights as a matched unit.

**Azure Static Web Apps unavailable** — Static Web Apps only deploys to five fixed regions, none of which overlap with this subscription's allowed regions. The frontend was instead containerized with nginx and deployed as a second Container App, matching the backend's deployment pattern.

**Ingress port configuration** — Azure Container Apps requires the ingress target port and container health probe ports to match the application's actual listening port. Mismatches cause silent "Activation failed" errors with no clear log output. The nginx frontend listens on port 80; the FastAPI backend on port 7860.

**Docker build context** — `.dockerignore` excludes `node_modules` to prevent locally-installed Windows binaries from overwriting the correct Linux versions built inside the container, a subtle source of runtime failures when building on Windows for Linux containers.

---

## Disclaimer

This tool is intended for research and educational purposes only. It is not a substitute for professional medical diagnosis. Always consult a qualified medical professional for any health concerns.
