# 🏠 House Price Prediction App

A machine learning project to predict house prices using the **Ames Housing Dataset**.  
The project explores different regression algorithms and deploys a **Streamlit web app** for interactive predictions.

---

## 🚀 Features

- Data preprocessing and cleaning for reliable predictions
- Multiple regression models trained and compared (Linear Regression, Random Forest, XGBoost)
- Model evaluation with metrics (RMSE, R²)
- Interactive **Streamlit app** for real-time predictions
- Visualization of model performance and feature importance

---

## 📊 Models Used

- **Linear Regression** – Baseline model
- **Random Forest Regressor** – Handles non-linearity and feature interactions
- **XGBoost Regressor** – High-performance boosting algorithm

---

## 📈 Results

Best model achieved:

- **XGBoost** with **RMSE ≈ 0.149** and **R² ≈ 0.881**

---

## 🖥️ Streamlit App

The web app allows users to:

- Input house features (overall quality, area, garage size, year built, etc.)
- Get an estimated house price prediction instantly
- View the most important features influencing the prediction

Run the app locally:

```bash
streamlit run streamlit_app.py
```

---

## 🛠️ Tech Stack

- Python (pandas, numpy, scikit-learn, xgboost)
- Streamlit for the web interface
- Matplotlib & Seaborn for visualization
- Joblib for model persistence

---

## 📂 Data

- **Dataset** : Ames Housing Dataset
  Includes housing details such as overall quality, living area, year built, and sale price.

## ⚡ How to Run

- Clone this repository:

```bash
git clone https://github.com/your-username/HousePricePrediction.git
cd HousePricePrediction
```

- Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

- Install dependencies:

```bash
pip install -r requirements.txt
```

- Run tests to verify setup:

```bash
pytest -q
```

- Launch the Streamlit app:

```bash
streamlit run streamlit_app.py
```

---

## 📌 Future Improvements

- Deploy app to Heroku or Streamlit Cloud
- Add more features from the dataset
- Experiment with hyperparameter tuning for better performance
- Support ensemble predictions

---

## 📝 License

This project is licensed under the MIT License.
