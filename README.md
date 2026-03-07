# Credit Scoring Model
### Credit Models project

**Author:** 
  - José Armando Melchor Soto
  - Rolando Fortanell Canedo
  - David Campos Ambriz

**Course:** Credit Models  
**Institution:** ITESO  

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
  - [Functional Architecture](#functional-architecture)
  - [OOP Architecture](#oop-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Details](#pipeline-details)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Feature Set](#2-feature-set)
  - [3. Model Training](#3-model-training)
  - [4. Credit Scoring](#4-credit-scoring)
  - [5. Evaluation & Visualization](#5-evaluation--visualization)
- [Output](#output)
- [License](#license)

---

## Overview

The **Credit Scoring Model** transforms raw consumer credit data into interpretable credit scores and risk categories. The end-to-end workflow covers:

1. **Data cleaning** – removes PII columns, strips noise, imputes missing values, and enforces logical ranges.
2. **Feature engineering** – one-hot encodes categoricals and selects the nine most predictive features.
3. **Model training** – fits a balanced `LogisticRegression` inside a `StandardScaler` → model `sklearn` Pipeline.
4. **Scorecard construction** – derives a linear scorecard from the difference between the "Good" and "Poor" logit coefficients.
5. **Scoring & classification** – rescales the linear score to 0–500 and classifies each record as *Poor*, *Standard*, or *Good* using fixed thresholds.
6. **Visualizations** – generates confusion matrices, score-distribution plots, and real-vs-predicted KDE charts.

---


## Architecture

### Project Structure

``` mermaid
flowchart TD
    ROOT["Credit_Scoring_Model/"]

    ROOT --> DATA["data/"]
    ROOT --> NOTEBOOKS["notebooks/"]
    ROOT --> SRC["src/"]
    ROOT --> MAIN["main.py"]
    ROOT --> REQ["requirements.txt"]
    ROOT --> README["README.md"]

    DATA --> D1["train-3.csv"]
    D1 --> D2["clean_train.csv"]
    D2 --> D3["scores_full_dataset.csv"]

    NOTEBOOKS --> N1["feature_analysis.ipynb"]
    N1 --> N2["test.ipynb"]

    SRC --> SRC_DATA["data/"]
    SRC --> SRC_MOD["modeling/"]
    SRC --> SRC_UTILS["utils/"]
    SRC --> SRC_VIZ["visualization/"]

    SRC_DATA --> SD1["data_cleaning.py"]
    SD1 --> SD2["load_data.py"]

    SRC_MOD --> SM1["train_model.py"]
    SM1 --> SM2["trainer.py"]
    SM2 --> SM3["score_model.py"]
    SM3 --> SM4["score_pipeline.py"]

    SRC_UTILS --> SU1["utils.py"]

    SRC_VIZ --> SV1["viz.py"]

    style ROOT fill:#222,stroke:#222,color:#fff,font-weight:bold
    style DATA fill:#444,stroke:#444,color:#fff
    style NOTEBOOKS fill:#444,stroke:#444,color:#fff
    style SRC fill:#444,stroke:#444,color:#fff
    style MAIN fill:#444,stroke:#444,color:#fff
    style REQ fill:#444,stroke:#444,color:#fff
    style README fill:#444,stroke:#444,color:#fff

    style SRC_DATA fill:#666,stroke:#666,color:#fff
    style SRC_MOD fill:#666,stroke:#666,color:#fff
    style SRC_UTILS fill:#666,stroke:#666,color:#fff
    style SRC_VIZ fill:#666,stroke:#666,color:#fff

    style D1 fill:#999,stroke:#999,color:#fff
    style D2 fill:#999,stroke:#999,color:#fff
    style D3 fill:#999,stroke:#999,color:#fff
    style N1 fill:#999,stroke:#999,color:#fff
    style N2 fill:#999,stroke:#999,color:#fff
    style SD1 fill:#999,stroke:#999,color:#fff
    style SD2 fill:#999,stroke:#999,color:#fff
    style SM1 fill:#999,stroke:#999,color:#fff
    style SM2 fill:#999,stroke:#999,color:#fff
    style SM3 fill:#999,stroke:#999,color:#fff
    style SM4 fill:#999,stroke:#999,color:#fff
    style SU1 fill:#999,stroke:#999,color:#fff
    style SV1 fill:#999,stroke:#999,color:#fff

```



### Functional Architecture

Data-flow from raw CSV to scored output:

```mermaid
flowchart TD
    subgraph INPUT["Input"]
        RAW["raw_train.csv\ntrain-3.csv"]
    end

    RAW -->|"raw DataFrame"| DC

    subgraph SRCDATA["src/data"]
        DC["data_cleaning.py\ndata_preprocessing()"]
        DC --> CLEAN["clean_train.csv"]
        CLEAN --> LD["load_data.py\nload_data()"]
    end

    LD -->|"X: DataFrame\ny: Series"| MAIN["main.py\nmain()"]

    MAIN -->|"X_train, y_train\nX_test"| TSM

    subgraph SRCMOD["src/modeling"]
        TSM["score_pipeline.py\ntrain_score_model()"]
        TSM -->|"model,\nX_train_scaled\nX_test_scaled"| EVM["score_pipeline.py\nevaluate_model()"]
        SDS["score_pipeline.py\nscore_dataset()"]
    end

    MAIN -->|"X full, model\nexperiment"| SDS

    EVM -->|"y_true, y_pred"| CM
    EVM -->|"scores, labels\nthresholds"| SD
    EVM -->|"scores, real\npredicted"| RVP
    EVM -->|"acc, scores\ny_pred"| PR

    subgraph SRCVIZ["src/visualization"]
        CM["viz.py\nplot_confusion_matrix()"]
        SD["viz.py\nplot_score_distribution()"]
        RVP["viz.py\nplot_real_vs_predicted()"]
    end

    subgraph SRCUTILS["src/utils"]
        PR["utils.py\nprint_results()"]
    end

    subgraph OUTPUT["Output"]
        CSV["scores_full_dataset.csv"]
        PLOTS["Plots / Charts"]
    end

    SDS --> CSV
    CM --> PLOTS
    SD --> PLOTS
    RVP --> PLOTS
    PR --> PLOTS

    %% Input
    style INPUT fill:#2a2a2a,stroke:#888888,color:#ffffff
    style RAW fill:#1a1a1a,stroke:#777777,color:#dddddd

    %% src/data
    style SRCDATA fill:#2f2f2f,stroke:#aaaaaa,stroke-width:2px,color:#ffffff
    style DC fill:#1a1a1a,stroke:#888888,color:#dddddd
    style CLEAN fill:#1a1a1a,stroke:#888888,color:#dddddd
    style LD fill:#1a1a1a,stroke:#888888,color:#dddddd

    %% main
    style MAIN fill:#333333,stroke:#cccccc,stroke-width:2px,color:#ffffff,font-weight:bold

    %% src/modeling
    style SRCMOD fill:#222222,stroke:#aaaaaa,stroke-width:2px,color:#ffffff
    style TSM fill:#1a1a1a,stroke:#888888,color:#dddddd
    style EVM fill:#1a1a1a,stroke:#888888,color:#dddddd
    style SDS fill:#1a1a1a,stroke:#888888,color:#dddddd

    %% src/visualization
    style SRCVIZ fill:#2a2a2a,stroke:#aaaaaa,stroke-width:2px,color:#ffffff
    style CM fill:#1a1a1a,stroke:#888888,color:#dddddd
    style SD fill:#1a1a1a,stroke:#888888,color:#dddddd
    style RVP fill:#1a1a1a,stroke:#888888,color:#dddddd

    %% src/utils
    style SRCUTILS fill:#333333,stroke:#aaaaaa,stroke-width:2px,color:#ffffff
    style PR fill:#1a1a1a,stroke:#888888,color:#dddddd

    %% output
    style OUTPUT fill:#2a2a2a,stroke:#aaaaaa,stroke-width:2px,color:#ffffff
    style CSV fill:#1a1a1a,stroke:#888888,color:#dddddd
    style PLOTS fill:#1a1a1a,stroke:#888888,color:#dddddd
```

### Object-Oriented Architecture

Class diagram showing all components and their relationships:

```mermaid
classDiagram
    direction TB

    class CreditScoringPipeline {
        +model : sklearn estimator
        +scale_numeric : bool
        +pipeline : Pipeline
        ---
        +_build_preprocessor(X) ColumnTransformer
        +build_pipeline(X) None
        +cross_validate(X, y, cv) tuple~float,float~
        +fit(X, y) None
        +evaluate(X_test, y_test) tuple~float,float,str~
        +predict(X) ndarray
        +get_coefficients() tuple~ndarray,ndarray~
        +save(path) None
    }

    class Experiment {
        +version : str
        +model : LogisticRegression
        +pipeline : CreditScoringPipeline
        +coef : ndarray
        +intercept : ndarray
        ---
        +split_data(X, y) tuple
        +run(X_train, y_train) tuple~ndarray,ndarray~
        +transform(X) ndarray
    }

    class CreditScoreModel {
        +coef : ndarray
        +intercept : float
        +features : list~str~
        +t1 : int = 327
        +t2 : int = 409
        ---
        +score(X) ndarray
        +credit_score(X) ndarray
        +predict(X) ndarray
        +classify(score) int
    }

    class ScorePipeline {
        <<module: score_pipeline>>
        +train_score_model(experiment, X_train, y_train, X_test, features) tuple
        +evaluate_model(model, X_train, X_test, y_test) tuple
        +score_dataset(data, model, experiment, save_path) DataFrame
    }

    class DataModule {
        <<module: data_cleaning + load_data>>
        +FINAL_FEATURES : list~str~
        +DROP_COLS : list~str~
        +RANGE_RULES : list~tuple~
        ---
        +data_preprocessing(data, save_path) DataFrame
        +load_data(path) tuple~DataFrame,Series~
        +_convert_credit_history_age(x) float
        +_apply_range_rule(data, col, min, max) DataFrame
        +_impute_numeric(data, col) DataFrame
    }

    class VizModule {
        <<module: viz>>
        ---
        +plot_confusion_matrix(y_true, y_pred, class_names, model_name) None
        +plot_score_distribution(scores, labels, thresholds, dataset_name) None
        +plot_real_vs_predicted(scores, true_labels, pred_labels, dataset_name) None
    }

    class UtilsModule {
        <<module: utils>>
        ---
        +print_results(acc, scores, model) None
    }

    class Main {
        <<entrypoint: main.py>>
        ---
        +main() None
    }

    %% Relationships
    Experiment "1" --> "1" CreditScoringPipeline : creates & owns
    Experiment "1" --> "1" CreditScoreModel : provides coef/intercept to
    ScorePipeline ..> Experiment : uses
    ScorePipeline ..> CreditScoreModel : creates
    Main ..> DataModule : calls load_data()
    Main ..> Experiment : instantiates
    Main ..> ScorePipeline : calls train / evaluate / score
    Main ..> VizModule : calls plots
    Main ..> UtilsModule : calls print_results()
    CreditScoringPipeline ..> CreditScoreModel : supplies coef via get_coefficients()
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/ppmelch/Credit_Scoring_Model.git
cd Credit_Scoring_Model

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Step 1 – Preprocess the raw data

Run this once to generate `data/clean_train.csv` from the raw file:

```python
import pandas as pd
from src.data.data_cleaning import data_preprocessing

raw = pd.read_csv("data/train-3.csv")
data_preprocessing(raw, save_path="data/clean_train.csv")
```

### Step 2 – Train, evaluate and score

```bash
python main.py
```

`main.py` will:
- Load `data/clean_train.csv`
- Train a logistic-regression scorecard
- Print accuracy and threshold information
- Display confusion matrices and distribution plots
- Save full-dataset scores to `data/scores_full_dataset.csv`

---

## Pipeline Details

### 1. Data Preprocessing

`src/data/data_cleaning.py` – `data_preprocessing()`

| Step | Description |
|------|-------------|
| Drop PII columns | Removes `ID`, `Customer_ID`, `Month`, `Name`, `SSN` |
| Strip noise | Removes `_` and `!@9#%8` patterns via regex |
| Numeric conversion | Strips non-numeric characters from noisy float fields |
| Credit History Age | Converts `"22 Years and 3 Months"` → `22.25` |
| Categorical imputation | Fills missing values in `Credit_Mix`, `Occupation`, `Type_of_Loan`, `Payment_Behaviour` with `"Missing"` |
| Range validation | Flags and nullifies out-of-range values (e.g. `Age` must be 18–100) |
| Numeric imputation | Fills remaining NaN values with the column median; adds `_missing` indicator columns |
| Normalize categories | Converts `"NM"` in `Payment_of_Min_Amount` to `"No"` |

### 2. Feature Set

Nine features are used for modeling (selected after one-hot encoding):

| Feature | Type |
|---------|------|
| `Outstanding_Debt` | Numeric |
| `Interest_Rate` | Numeric |
| `Delay_from_due_date` | Numeric |
| `Num_Credit_Card` | Numeric |
| `Changed_Credit_Limit` | Numeric |
| `Total_EMI_per_month` | Numeric |
| `Credit_Mix_Standard` | Dummy |
| `Credit_Mix_Good` | Dummy |
| `Payment_of_Min_Amount_Yes` | Dummy |

### 3. Model Training

`src/modeling/trainer.py` – `Experiment`  
`src/modeling/train_model.py` – `CreditScoringPipeline`

- An 80/20 **stratified train–test split** is applied.
- A `StandardScaler → LogisticRegression` sklearn `Pipeline` is fitted.
- `LogisticRegression` uses `class_weight="balanced"` and `max_iter=10000`.
- The scorecard coefficients are derived as:

  ```
  coef_score  = coef[Good]  − coef[Poor]
  intercept_score = intercept[Good] − intercept[Poor]
  ```

### 4. Credit Scoring

`src/modeling/score_model.py` – `CreditScoreModel`

The linear score is computed and min-max normalized to 0–500:

```
z     = X · coef_score + intercept_score
score = (z − z_min) / (z_max − z_min) × 500
```

Classification thresholds (scores are integers):

| Score range | Category | Condition |
|-------------|----------|-----------|
| 0 – 326 | **Poor** (class 0) | `score < 327` |
| 327 – 408 | **Standard** (class 1) | `327 ≤ score < 409` |
| 409 – 500 | **Good** (class 2) | `score ≥ 409` |

### 5. Evaluation & Visualization

`src/modeling/score_pipeline.py` – `evaluate_model()`  
`src/visualization/viz.py`

- **Confusion matrices** for train and test sets.
- **Score-distribution KDE plots** (overall and per class) with threshold markers.
- **Real vs. Predicted KDE plots** per credit category.

---

## Output

After running `main.py`, the file `data/scores_full_dataset.csv` is created with three columns:

| Column | Description |
|--------|-------------|
| `Score` | Normalized credit score (0–500) |
| `Classification` | Numeric class label (0, 1, 2) |
| `Credit_Category` | Human-readable label (`Poor`, `Standard`, `Good`) |

---

## About the Author

This repository was created by 
- **José Armando Melchor Soto**
- **Rolando Fortanell Canedo**
- **David Campos Ambriz** 

---
## License

This project is licensed under the **MIT License** - [LICENSE](LICENSE). 
