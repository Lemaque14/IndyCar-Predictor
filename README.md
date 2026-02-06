# IndyCar Streamlit Prediction App

Interactive Streamlit application that loads a trained machine learning model to predict IndyCar-related outcomes using historical data.
The model was trained using PyCaret with a CatBoost estimator and is executed locally via Streamlit.

# How to use it
### 1). Clone the repo
### 2). Create a virtual environment
## Windows ##
python -m venv .venv
.\.venv\Scripts\activate
## Mac / Linux
python3 -m venv
source .venv/bin/activate
### 3). Install dependencies
pip install -r requirements.txt
### 4). Run the streamlit app
streamlit run streamlit_indycar_predictor.py

