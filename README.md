# SHAP-LLM Shiny (Python) App

An interactive **Shiny for Python** app that demonstrates estimand-aware modeling and model interpretability using:
- **XGBoost / baseline models**
- **SHAP** (global + local explanations)
- **LIME** (local explanations)
- Optional **LLM-assisted explanations** (via OpenAI API key provided at runtime)

> **Note:** This app runs locally by default. To share it with others in your organization, it must be deployed (e.g., Posit Connect, Azure, internal server).

---

## 1) Project Structure

A typical folder layout:


- `app.py`: main Shiny app
- `requirements.txt`: Python dependencies
- `outputs/`: where generated SHAP/LIME figures may be written

---

## 2) Prerequisites

- **Python 3.10+** (recommended)
- pip installed
- (Optional) Git

Check your Python version:

```bash
python --version

3) Setup (Recommended: Virtual Environment)
Windows (PowerShell)

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
macOS / Linux
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
4) Run the App (Local)
shiny run app.py
5) OpenAI API Key (Optional)

If the app includes an LLM explanation feature, you typically have two safe options:

Option A: Enter the key in the app UI

If the app has a secure password input field, users can paste their key at runtime.

Option B: Use an environment variable (recommended)

Set the key in your terminal before running:

Windows (PowerShell)
$env:OPENAI_API_KEY="YOUR_KEY_HERE"
shiny run app.py

macOS / Linux
export OPENAI_API_KEY="YOUR_KEY_HERE"
shiny run app.py


Do NOT hardcode API keys in app.py or commit them to GitHub.

6) Common Issues & Fixes
A) ModuleNotFoundError: shiny

You didn’t install dependencies into the active environment.

Fix:

pip install -r requirements.txt

B) SHAP / matplotlib errors on headless servers

If running on a server without display, ensure matplotlib uses a non-interactive backend (Agg).
(You can add this inside app.py before importing pyplot if needed.)

C) Port already in use

Change the port:

shiny run app.py --port 8010
