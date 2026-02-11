# SHAP-LLM Shiny (Python) App

An interactive **Shiny for Python** app that demonstrates estimand-aware modeling and model interpretability using:

- **XGBoost / baseline models**
- **SHAP** (global + local explanations)
- **LIME** (local explanations)
- Optional **LLM-assisted explanations** (via OpenAI API key provided at runtime)

> Note: This app runs locally by default. To share it with others in your organization, it must be deployed (e.g., Posit Connect, Azure, internal server).

---

## 1) Project Structure

A typical folder layout:

    shap-llm-shiny/
    │
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── www/
    │   └── (optional static assets)
    └── outputs/
        └── (generated SHAP/LIME figures)

- app.py — Main Shiny application  
- requirements.txt — Python dependencies  
- outputs/ — Folder where SHAP/LIME plots may be saved  

---

## 2) Prerequisites

- Python 3.10+ (recommended)
- pip installed
- (Optional) Git

Check your Python version:

    python --version

---

## 3) Setup (Recommended: Virtual Environment)

### Windows (PowerShell)

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt

### macOS / Linux

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

---

## 4) Run the App (Local)

After installing dependencies:

    shiny run app.py

You should see something like:

    Running on http://127.0.0.1:8000

Open that URL in your browser.

Run on a specific port:

    shiny run app.py --port 8000

Allow access from other machines on the same network (not recommended unless approved):

    shiny run app.py --host 0.0.0.0 --port 8000

Others would then use:

    http://<your-ip-address>:8000

This method is not secure and typically blocked by enterprise firewalls. Use proper deployment for production.

---

## 5) OpenAI API Key (Optional)

If the app includes an LLM explanation feature, you have two safe options:

### Option A: Enter the key in the app UI

If the app has a secure password input field, users can paste their key at runtime.

### Option B: Use an environment variable (Recommended)

Windows (PowerShell):

    $env:OPENAI_API_KEY="YOUR_KEY_HERE"
    shiny run app.py

macOS / Linux:

    export OPENAI_API_KEY="YOUR_KEY_HERE"
    shiny run app.py

Do NOT hardcode API keys in app.py or commit them to GitHub.

---

## 6) Common Issues & Fixes

A) ModuleNotFoundError: shiny  
You did not install dependencies in the active virtual environment.

Fix:

    pip install -r requirements.txt

B) Port Already in Use  
Run on a different port:

    shiny run app.py --port 8010

C) SHAP / matplotlib errors on headless servers  
If running on a server without display, ensure matplotlib uses a non-interactive backend (Agg).

Add near the top of app.py if needed:

    import matplotlib
    matplotlib.use("Agg")

---

## 7) Deployment (For Sharing Internally)

To share the app inside your organization, deploy using:

- Posit Connect
- Azure App Service
- Internal Docker / Kubernetes
- Internal Linux server

After deployment, map your internal short link (e.g., go/shapllm) to the deployed URL.

Do NOT map a short link to:

    http://127.0.0.1:8000

That only works locally.

---

## 8) Security Notes

- Never store API keys in source code
- Use environment variables
- Follow your organization’s IT security policies before deployment

---

## 9) Citation

- SHAP: Lundberg & Lee (2017)
- LIME: Ribeiro, Singh & Guestrin (2016)
- Shiny for Python: Posit

---

## 10) License

Add your preferred license here (e.g., MIT, Apache-2.0), or specify internal-use only.

