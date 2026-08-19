# 🛍️ Retail Demand Forecasting & Dynamic Price Optimization Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning pipeline and decision support platform that solves retail margin leakage and stockouts. By combining **unconstrained demand forecasting** with **constant price elasticity simulation**, this engine helps inventory and pricing managers optimize unit prices to maximize net profit while maintaining stock safety margins.

---

## 📌 Executive Summary & Key Features

Retailers often lose margin either by underpricing during high demand or overpricing price-sensitive items, leading to stockouts or excess inventory. This system addresses both sides of the coin:

* **Demand Forecasting Model**: Machine Learning pipeline using lagged features, rolling statistics, calendar seasonality, and promotional signals to forecast true, unconstrained demand.
* **Dynamic Price Elasticity Simulation**: Simulates non-linear customer demand response across price ranges to locate profit-maximizing price points.
* **Stockout & Risk Alerts**: Real-time comparison between predicted optimal demand and on-hand inventory levels to prevent revenue loss.
* **Executive Decision Dashboard**: An interactive Streamlit app designed for operational decision-making, offering what-if scenario analyses and model driver transparency.

---

## 📁 Repository Structure

```text
├── .github/
│   └── workflows/
│       └── ci_cd.yml               # GitHub Actions CI/CD workflow
├── config/
│   └── config.yaml                 # Central project configuration & hyperparams
├── data/
│   ├── raw/                        # Raw historical sales datasets (.gitkeep)
│   └── processed/                  # Cleaned, feature-engineered data (.gitkeep)
├── models/                         # Serialized trained model artifacts (.gitkeep)
├── reports/
│   └── figures/                    # Generated charts & analytical assets (.gitkeep)
├── src/
│   └── retail_inventory_ml/
│       ├── __init__.py
│       ├── api.py                  # API service layer (FastAPI/Flask integration)
│       ├── evaluate.py             # Model performance evaluation metrics
│       ├── feature_engineering.py  # Lag, rolling, & seasonality generation
│       ├── logger.py               # Centralized application logging
│       ├── preprocessing.py        # Data cleaning & encoding routines
│       ├── train.py                # Model training execution pipeline
│       └── utils.py                # Common helper scripts & file I/O
├── tests/
│   ├── __init__.py
│   ├── test_api.py                 # API integration tests
│   └── test_features.py            # Unit tests for feature transformers
├── Dockerfile                      # Production container deployment config
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # Python package dependencies
├── setup_env.sh                    # Environment setup script
├── streamlit_app.py                # Interactive Streamlit Command Center
└── template.py                     # Automated project structure builder
