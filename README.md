# 🏁 F1 Lap Time Predictor (Model Evaluation)

This project demonstrates a machine learning model trained to predict Formula 1 lap times using historical race data. The current version displays the **Mean Absolute Error (MAE)** to evaluate the model's performance.

## 🚀 Features

- 📊 Displays **Mean Absolute Error (MAE)** of the trained model
- 🧠 Model: Random Forest Regressor
- 🗃️ Data: Historical F1 race data (drivers, races, laps, etc.)
- 🌐 Frontend: HTML + TailwindCSS
- 🖥️ Backend: Flask API (Python)
- 📉 Future support for full lap-wise predictions and graphs

---

## 🗂️ Project Structure

f1-lap-prediction/
│
├── backend/
│ ├── app.py # Flask backend (returns MAE)
│ ├── model.pkl # Trained ML model
│ ├── trained_columns.pkl # Feature columns used for prediction
│ └── requirements.txt # Python dependencies
│
├── frontend/
│ └── index.html # Frontend for displaying MAE
