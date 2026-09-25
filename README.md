# SHAP-LLM Shiny (Python) App

An interactive **Shiny for Python** app that demonstrates estimand-aware modeling and model interpretability using:

- **XGBoost, random forest, and parametric baselines** (linear, logistic, Cox)
- **Original (clinical), A-learning and W-learning objectives** for continuous, binary and time-to-event outcomes
- **SHAP** (global + local explanations)
- **LIME** (local explanations)
- Optional **LLM-assisted explanations**, using either the hosted OpenAI API or a **locally deployed** open-weight model

This repository accompanies the tutorial *A Practical Tutorial on Interpreting Machine Learning Models in Precision Medicine Using SHAP, LIME, and Large Language Models* (Liu & Huang).

> Note: This app runs locally by default. To share it with others in your organization, it must be deployed (e.g., Posit Connect, Azure, internal server).

---

## Versions

| Folder | Version | Description |
| --- | --- | --- |
| `V1/` | **Revised (recommended)** | The revised app with all the new features listed below. |
| `/` (repository root) | Original | The app as it accompanied the original submission. Kept unchanged so the original results can be reproduced. |

The two versions are independent: each has its own `app.py` and `requirements.txt`.

---

## What's New in the Revised App (`V1/`)

### Modeling

- **Hyperparameter tuning.** Random or grid search with cross-validation on the training split only, so test metrics stay out-of-sample. Tuning is *objective-aware*: under A-/W-learning, configurations are ranked by the out-of-fold value of that same objective, not by RMSE or log-loss. The propensity score is re-estimated within each fold. The search space can be edited under **Hyperparameters and tuning → Advanced**.
- **Random Forest** added as a model type, for the Original objective only. A-/W-learning need a gradient interface that a forest doesn't have, so the app refuses those combinations and gives the reason, rather than silently fitting them.
- **Coefficient summaries** (new **Coefficients** tab) for linear, logistic and Cox models: estimate, SE, 95% interval, p-value and odds/hazard ratio. Where conventional inference is invalid (penalised fits, A-/W-learning), the app refuses the analytic interval and offers a **subject-level bootstrap** instead. The bootstrap refits the whole pipeline, including the propensity model, in each replicate. The tab also checks that SHAP values reproduce the coefficients for linear models.
- **Correctness fixes** in the time-to-event ITR models: the W-learning Cox gradient/Hessian weighting, the sign of the benefit score (now reported as −(Cox linear predictor), so larger always means greater benefit), and the `survival:cox` label encoding for censored rows. Also fixed: Logistic Regression with penalty 0 failing from the UI, and class selection for SHAP values of tree classifiers.

### Data preprocessing

- New **Data check** tab summarising missingness, scale and variable types.
- Explicit **missing-data policy**: complete cases, median imputation (fitted on the training split), or keeping missing values (XGBoost with the Original loss only).
- Optional **standardisation** (z-score, fitted on the training split only).
- LIME now declares categorical features and uses a **fixed perturbation seed**, so its output is reproducible from run to run.

### Explanation quality and sensitivity analysis

A new **Sensitivity analysis** panel (in the sidebar) and **Sensitivity** tab, with six modes:

| Mode | What it measures |
| --- | --- |
| LIME stability | How much LIME explanations move across seeds, `num_samples` and kernel widths |
| LLM temperature sweep | Narrative stability and faithfulness across temperatures, using the manuscript's Section 5 metrics |
| Both | The two above together |
| Template baseline vs LLM | Compares the LLM with a deterministic report template, graded by the same harness |
| Audience adaptation | Whether the LLM genuinely adapts to different audiences, using within-audience variation as the noise control |
| Attributions across models | Fits each model family on the identical split and compares attribution **ranks and shares** (Spearman correlation, top-K overlap) |

### LLM layer

- **Locally deployed models.** Any OpenAI-compatible server (vLLM, llama.cpp, Ollama, LM Studio) can be used by entering its base URL. No API key is needed, and nothing leaves the machine.
- **Model comparison.** Enter several comma-separated model names to grade them side by side on the identical payload and prompt.
- **Temperature control** in the sidebar (default 0, the reproducible setting).
- **Discussion tab.** Ask follow-up questions about the loaded explanation. Each answer uses the same data as the narrative and is graded for faithfulness, with the grade shown beside it.
- **Privacy statement.** Before each run, the sidebar states exactly what will be sent and where. There is also an option to round individual values in Local scope (see [Security Notes](#9-security-notes)).
- **Robustness.**
  - The app checks that the endpoint is reachable before the first request.
  - Requests time out after 180 s, with one retry.
  - `<think>` reasoning traces from local reasoning models are removed before grading.
  - Models that refuse an explicit temperature are detected, and the run is labelled accordingly.

### Exporting results

- New **Export results** panel with these downloads:
  - SHAP attributions, LIME weights and coefficients as CSV
  - A run manifest as JSON, recording configuration, seeds, tuning outcome and package versions
  - A single **All results (.zip)** that also includes the metrics and figures

  Previously the app saved only PNG figures to the server's `outputs/` folder. See [section 6](#6-exporting-results-for-downstream-analysis).

---

## 1) Project Structure

    shap-llm-shiny/
    │
    ├── app.py               # Original Shiny application
    ├── requirements.txt     # Dependencies for the original app
    ├── README.md
    ├── outputs/             # Generated SHAP/LIME figures
    └── V1/                  # Revised version of the app (recommended)
        ├── app.py
        └── requirements.txt

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
    pip install -r V1/requirements.txt

### macOS / Linux

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r V1/requirements.txt

For the original app, install from the top-level `requirements.txt` instead. The revised app also needs `scipy` and `lifelines`, which are both listed in `V1/requirements.txt`.

**Optional:** semantic similarity between generated narratives uses `all-MiniLM-L6-v2` when `sentence-transformers` is installed. Otherwise it falls back to TF-IDF cosine, and the output reports the fallback. It pulls in PyTorch, so it isn't installed by default:

    pip install sentence-transformers

---

## 4) Run the App (Local)

Revised app (recommended):

    shiny run V1/app.py

Original app:

    shiny run app.py

You should see something like:

    Running on http://127.0.0.1:8000

Open that URL in your browser.

Run on a specific port:

    shiny run V1/app.py --port 8000

Allow access from other machines on the same network (not recommended unless approved):

    shiny run V1/app.py --host 0.0.0.0 --port 8000

Others would then use:

    http://<your-ip-address>:8000

This method is not secure and is typically blocked by enterprise firewalls. Use proper deployment for production.

### Quick start (revised app)

1. Upload a CSV file.
2. Choose the outcome type, model and objective (e.g., XGBoost + A-learning).
3. Press **Run analysis**, then browse the Metrics, Coefficients, SHAP plots, LIME plots and LLM explanation tabs.
4. Optionally, open **Sensitivity analysis** in the sidebar and press **Run sensitivity analysis**.
5. Download everything from **Export results → All results (.zip)**.

---

## 5) LLM Endpoint: Hosted API or a Locally Deployed Model

*(Option B and model comparison are available in the revised app only.)*

The narrative step can run against either a hosted API or a local model. You choose in the sidebar under **LLM explanation settings → Endpoint and privacy**. Before you press Run, that panel also states exactly what the current configuration will send and where.

### Option A: Hosted OpenAI API

Set the key in the environment and leave the endpoint field blank.

Windows (PowerShell):

    $env:OPENAI_API_KEY="YOUR_KEY_HERE"
    shiny run V1/app.py

macOS / Linux:

    export OPENAI_API_KEY="YOUR_KEY_HERE"
    shiny run V1/app.py

Do NOT hardcode API keys in app.py or commit them to GitHub.

### Option B: A locally deployed model (no key, nothing leaves the machine)

Serve any OpenAI-compatible endpoint on this machine and put its base URL in the **Endpoint (base URL)** field. No `OPENAI_API_KEY` is needed. Examples:

    # vLLM
    python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8000
    #   -> endpoint http://localhost:8000/v1 , model Qwen/Qwen2.5-7B-Instruct

    # llama.cpp
    llama-server -m ./model.gguf --port 8080
    #   -> endpoint http://localhost:8080/v1

    # Ollama
    ollama serve && ollama pull qwen2.5:7b
    #   -> endpoint http://localhost:11434/v1 , model qwen2.5:7b

    # LM Studio: start the local server from the Developer tab
    #   -> endpoint http://localhost:1234/v1

Set **Model** to whatever name the server exposes. You can also supply the endpoint through the `LLM_BASE_URL` environment variable.

**Note:** if the local LLM server uses port 8000, run the Shiny app on a different port (e.g., `shiny run V1/app.py --port 8010`).

Before the first request, the app checks that something is answering at the endpoint. If the server isn't running, you get an immediate message naming the problem. Requests then use a 180-second timeout with one retry.

Generation blocks the UI while it runs. On CPU, expect roughly 20–60 seconds per narrative. A sensitivity sweep makes generations × temperatures × models requests, so size it accordingly.

Local servers implement `/v1/chat/completions` but generally not `/v1/responses`. The app detects which one is available on first use, so nothing needs configuring. It also detects a model that refuses an explicit `temperature`: it retries without it and labels the run, so a sweep where the temperature was ignored is reported as such.

There are two reasons to prefer a local model beyond cost:

- Its weights are fixed, so results are reproducible in a way a periodically updated hosted model is not.
- It accepts `temperature`, which some hosted reasoning models reject.

### Comparing models

Enter two or more comma-separated names in **Sensitivity analysis → Model(s)** to grade them side by side on the identical scenario, payload and prompt (e.g., a hosted model against a local one). If the endpoint doesn't serve one of the models, that model is reported as failed and the others' results are kept.

---

## 6) Exporting Results for Downstream Analysis

*(Revised app only.)* Open **Export results** in the sidebar. Five downloads:

| Download | Contents |
| --- | --- |
| All results (.zip) | Everything below, plus the figures and a `README.txt` |
| SHAP attributions (.csv) | Tidy long form: one row per (observation, feature) with the observed covariate value and its SHAP value |
| LIME weights (.csv) | Mean absolute LIME weight per feature |
| Coefficients (.csv) | Estimate, standard error, interval, p-value, odds/hazard ratio; bootstrap columns where a conventional interval is not valid |
| Run manifest (.json) | Configuration, seeds, tuning outcome, preprocessing, and the version of every package that can change a number |

The zip also contains:

- The wide SHAP matrix and per-feature importances
- Metrics
- Any sensitivity, template-baseline or audience-adaptation results
- The cross-model comparison
- The LLM narratives with their faithfulness grades

**Read the manifest before reusing the numbers.** A CSV of attributions is not a reproducible result without a record of the model, objective, seed and package versions that produced it.

**Comparing attributions between model families.** A SHAP value carries the units of the model's own output:

- log-odds for XGBoost `binary:logistic`
- probability for a random forest classifier
- outcome units for a linear model

Raw magnitudes are therefore *not* comparable across families. Every export includes dimensionless `share_of_total` and `rank` columns; compare those instead.

---

## 7) Common Issues & Fixes

A) ModuleNotFoundError: shiny  
You did not install dependencies in the active virtual environment.

Fix:

    pip install -r V1/requirements.txt

B) Port Already in Use  
Run on a different port:

    shiny run V1/app.py --port 8010

C) SHAP / matplotlib errors on headless servers  
If running on a server without display, ensure matplotlib uses a non-interactive backend (Agg).

Add near the top of app.py if needed:

    import matplotlib
    matplotlib.use("Agg")

D) "OPENAI_API_KEY is not set"  
Either set the key (section 5, Option A) or enter a local endpoint (Option B). In the revised app, the key is only required when no endpoint is given.

E) A model / objective combination is refused  
Some combinations are not statistically meaningful, for example Random Forest with A-/W-learning, or analytic coefficient intervals for a penalised fit. The app explains why and, where possible, offers an alternative such as the bootstrap.

---

## 8) Deployment (For Sharing Internally)

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

## 9) Security Notes

- Never store API keys in source code
- Use environment variables
- Follow your organization's IT security policies before deployment

### What the LLM step transmits

Credentials are not the only consideration. The prompt itself carries data:

- **Global** explanation scope sends feature names and mean absolute attributions only: aggregates over the explanation set, with no individual records.
- **Local** explanation scope sends the selected individual's **observed covariate values**, at full precision, along with their attributions. With a hosted API, those values leave the machine.

Mitigations, strongest first:

1. Deploy the model locally (section 5, Option B).
2. Use Global scope.
3. Set **Individual values sent in Local scope** to *Round to 3 significant figures*.

In the revised app, the sidebar states which of these applies before each run. Confirm the configuration against your organization's policy before using real patient data.

---

## 10) Citation

If you use this app, please cite the accompanying tutorial:

- Liu & Huang. *A Practical Tutorial on Interpreting Machine Learning Models in Precision Medicine Using SHAP, LIME, and Large Language Models.* (Citation details to be added on publication.)

Methods and software:

- SHAP: Lundberg & Lee (2017)
- LIME: Ribeiro, Singh & Guestrin (2016)
- Shiny for Python: Posit

---

## 11) License

Add your preferred license here (e.g., MIT, Apache-2.0), or specify internal-use only.


