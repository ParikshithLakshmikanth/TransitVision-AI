# TransitVision AI — Kaggle Data Access & Authentication Protocols

This document specifies the authentication, download, and verification protocols for accessing the **Kandy Bus Travel Time Dataset** from Kaggle.

---

## 1. Primary Dataset Overview

* **Dataset Title**: Bus Travel Time Data (Kandy, Sri Lanka)
* **Author / Provider**: Shiveswarran Ratneswaran & Dr. Uthayasanker Thayasivam (University of Moratuwa / University of Peradeniya)
* **Kaggle URL**: [https://www.kaggle.com/datasets/shiveswarranr/bus-travel-time-data](https://www.kaggle.com/datasets/shiveswarranr/bus-travel-time-data)
* **Licensing**: Open Research / Academic Access (Copyright Authors)
* **Temporal Span**: October 1, 2021 to November 1, 2022 (~9 months of bus probe observations)
* **Coverage**: Route 654 (Kandy $\leftrightarrow$ Digana) & Route 690 (Kandy $\leftrightarrow$ Kadugannawa)

---

## 2. Authentication Protocol (Non-Hardcoded)

TransitVision AI strictly enforces that API keys or credentials must never be committed to source code.

### Supported Authentication Methods:

#### Option A: Kaggle Environment Variables (Recommended for Automated Pipelines)
Supply credentials at runtime via shell environment:
```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_api_token"
```

#### Option B: Standard API Token (`kaggle.json`)
Place the official Kaggle API JSON token under the user home configuration directory:
* **Linux/macOS**: `~/.kaggle/kaggle.json`
* **Windows**: `%USERPROFILE%\.kaggle\kaggle.json`
* **File Permissions**: Ensure read permissions are restricted (`chmod 600 ~/.kaggle/kaggle.json`).

---

## 3. Automated Ingestion & Extraction Workflow

The ingestion script [`scripts/download_kandy_data.py`](file:///d:/Parikshith/College/PBL/ML/TransitVision/Prj/scripts/download_kandy_data.py) performs:
1. Cryptographic check of existing raw archives in `data/raw/kandy_bus/`.
2. Execution of Kaggle API dataset download if files are missing.
3. Fetching of corresponding historical weather observations from Open-Meteo for Kandy coordinates ($7.2906^{\circ}\text{N}, 80.6337^{\circ}\text{E}$).
4. Calculation and serialization of SHA-256 checksums in `data/metadata/kandy_download_manifest.json`.
