# 🎾 Tennis Match Winner Prediction using Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green?style=for-the-badge)
![CatBoost](https://img.shields.io/badge/CatBoost-Gradient%20Boosting-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

### Predicting ATP Tennis Match Winners using Feature Engineering & Ensemble Machine Learning

</div>

---

# 📌 Overview

This project builds an end-to-end Machine Learning pipeline capable of predicting the winner of ATP tennis matches using only **pre-match information**.

Unlike many basic classification projects, this project focuses heavily on **feature engineering** by constructing historical statistics for each player before every match while carefully avoiding **data leakage**.

The project compares multiple machine learning algorithms and evaluates their performance on both random and chronological train-test splits.

---

# 🚀 Features

- 📊 Extensive Data Preprocessing
- 🎯 Historical Feature Engineering
- 🎾 Surface-specific player statistics
- 🔥 Recent Form Calculation
- 🤝 Head-to-Head Statistics
- 📈 Rank-based Features
- 🧠 Multiple Machine Learning Models
- ⚙ Hyperparameter Tuning
- 📉 Model Comparison

---

# 📂 Dataset Features

The final dataset contains the following features:

| Feature | Description |
|----------|-------------|
| Court | Indoor / Outdoor |
| Surface | Hard / Clay / Grass / Carpet |
| Round | Tournament Round |
| Best Of | Best of 3 or 5 Sets |
| H2H_WinPct | Previous Head-to-Head Win Percentage |
| RecentWinPct_1 | Player 1 Recent Win Percentage |
| RecentWinPct_2 | Player 2 Recent Win Percentage |
| Surface_WinPct_1 | Player 1 Surface Win Percentage |
| Surface_WinPct_2 | Player 2 Surface Win Percentage |
| Season | Season of Tournament |
| Higher_Rank_Player | Higher Ranked Player |
| Abs_Rank_Diff | Absolute Ranking Difference |

Target:

```
Winner_Encoded
```

---

# 🧹 Feature Engineering

One of the major goals of this project was to create meaningful historical features.

### ✅ Recent Form

For every player,

- Previous 5 Matches
- Rolling Win Percentage

Example

```
Wins: W W L W W

Recent Win Percentage = 4 / 5 = 0.80
```

---

### ✅ Surface Win Percentage

Calculated separately for

- Hard Court
- Clay
- Grass
- Carpet

using only previous matches on the same surface.

---

### ✅ Head-to-Head Win Percentage

For every pair of players,

```
Player A Wins
-------------------------
Total Previous Meetings
```

was computed before the current match.

No future information was used.

---

### ✅ Ranking Features

Instead of raw rankings, engineered features were used.

- Absolute Rank Difference
- Higher Ranked Player

These capture both the magnitude and direction of ranking advantage.

---

# 🔒 Preventing Data Leakage

A major emphasis of this project was ensuring that no future information leaked into training features.

Historical statistics such as

- Recent Form
- Surface Win Percentage
- Head-to-Head Record

were updated **only after** processing each match.

This guarantees that each prediction uses only information that would have been available before the match was played.

---

# 🤖 Models Used

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- AdaBoost
- XGBoost
- CatBoost

---

# ⚙ Hyperparameter Tuning

Hyperparameter optimization was performed using

- GridSearchCV
- RandomizedSearchCV

to obtain improved model performance.

---

# 📊 Model Performance

| Model | Test Accuracy |
|--------|--------------:|
| Logistic Regression | **65.8%** |
| KNN | **63.7%** |
| AdaBoost | **66.7%** |
| CatBoost | **66.7%** |
| **XGBoost** | **66.9%** ⭐ |

Using a chronological train-test split, the best-performing model achieved approximately **65%** accuracy.

---

# 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- CatBoost

---

# 📁 Project Structure

```
Tennis Match Winner Predictor/
│
├── 📂 .vscode/                     # VS Code workspace settings
├── 📂 artifacts/                   # Saved preprocessing pipeline and trained model
├── 📂 catboost_info/               # CatBoost training logs
├── 📂 logs/                        # Application and training logs
├── 📂 ml_project.egg-info/         # Package metadata
├── 📂 Notebooks/                   # Jupyter notebooks for EDA & experimentation
│
├── 📂 src/
│   ├── 📂 components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py       # Data loading and train-test split
│   │   ├── data_transformation.py  # Feature engineering & preprocessing pipeline
│   │   └── model_trainer.py        # Model training and evaluation
│   │
│   ├── 📂 pipeline/
│   │   ├── __init__.py
│   │   └── predict_pipeline.py     # Prediction pipeline for Flask app
│   │
│   ├── __init__.py
│   ├── exceptions.py               # Custom exception handling
│   ├── logger.py                   # Logging configuration
│   └── utils.py                    # Helper functions
│
├── 📂 templates/
│   ├── home.html                   # Prediction page
│   └── index.html                  # Landing page
│
├── 📂 venv/                        # Virtual environment
│
├── app.py                          # Flask application entry point
├── requirements.txt                # Project dependencies
├── setup.py                        # Package installation configuration
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore rules
```

---

# ▶ Running the Project

Clone the repository

```bash
git clone https://github.com/yourusername/Tennis-Match-Winner-Predictor.git
```

Move into the directory

```bash
cd Tennis-Match-Winner-Predictor
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

_**Or Directly Use Deployed Link**_ </br></br>
**URL** :- https://preddtennis.onrender.com

---

# 📈 Future Improvements

- ATP Ranking Points
- Player Age
- Player Height
- Handedness
- Tournament Category
- Elo Rating
- Injury Information
- Betting Odds
- Streamlit Web Application
- SHAP Explainability

---

# 💡 Key Learnings

Through this project I gained hands-on experience with

- Data Cleaning
- Feature Engineering
- Rolling Statistics
- Historical Feature Construction
- Preventing Data Leakage
- Classification Models
- Hyperparameter Tuning
- Ensemble Learning
- Model Evaluation

---

# 👨‍💻 Author

## Abhinav Singh

**B.Tech**  
**Indian Institute of Technology Bhubaneswar (IIT Bhubaneswar)**

📌 Passionate about Machine Learning, Data Science, Artificial Intelligence, and Software Development.

---

<div align="center">

### ⭐ If you found this project interesting, consider giving it a Star!

Made with ❤️

<div align="center">

### 🎾 Built with Python, Machine Learning & a love for Tennis

</div>
