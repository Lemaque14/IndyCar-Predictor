import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# --- Helper Functions ---
def time_to_seconds(t: str) -> float:
    """Convert a string MM:SS.s to total seconds."""
    m, s = t.split(':')
    return int(m) * 60 + float(s)

# --- Load Model & Schema ---
@st.cache_resource
def load_model_and_schema(model_path: str, data_path: str):
    model = XGBRegressor()
    model.load_model(model_path)
    df = pd.read_csv(data_path)
    df['BestLapTimeSec'] = df['BestLapTime'].apply(time_to_seconds)
    X_cols = df.drop(columns=['BestLapTime', 'PositionFinish']).columns.tolist()
    return model, X_cols

model, FEATURE_COLS = load_model_and_schema(
    model_path="C:\\Users\\ripa_\\Downloads\\indycar_xgb.json",
    data_path='C:\\Users\\ripa_\\Downloads\\IndyCar Regression Data_v2.csv'
)

# --- Static Lists ---
TRACKS = ['BAR','BEL','GPI','IND','IOW','LB','MCO','MOL','POR','RUT','STP','TEX','TOR','VAN','WAT','WIS']
TEAMS  = [
    'Arrow McLaren','Chip Ganassi Racing','Dale Coyne Racing','Ed Carpenter Racing',
    'Juncos Hollinger Racing','Meyer Shank Racing','PREMA Racing',
    'Rahal Letterman Lanigan Racing','Team Penske'
]
DRIVERS = [
    'Alex Palou','Alexander Rossi','Callum Ilott','Christian Lundgaard','Colton Herta','Conor Daly',
    'David Malukas','Devlin DeFrancesco','Felix Rosenqvist','Graham Rahal','Josef Newgarden',
    'Kyffin Simpson','Kyle Kirkwood','Marcus Armstrong','Marcus Ericsson','Nolan Siegel',
    "Pato O'Ward",'Rinus VeeKay','Santino Ferrucci','Scott Dixon','Scott McLaughlin','Sting Ray Robb'
]
TRACKTYPE = ['R','O']

# Filter only those that match your feature cols
TRACK_OPTIONS = [t for t in TRACKS if t in FEATURE_COLS]
TEAM_OPTIONS  = [t for t in TEAMS  if t in FEATURE_COLS]
DRIVER_OPTIONS = [d for d in DRIVERS if d in FEATURE_COLS]
TRACKTYPE_OPTIONS = [tt for tt in TRACKTYPE if tt in FEATURE_COLS]

# --- Streamlit UI ---
st.title("IndyCar Race Result Predictor")
st.markdown("Fill in the fields below and click **Predict** to estimate finishing position.")

# Inputs
best_lap      = st.text_input("Best Lap Time (MM:SS.s)", "01:08.2")
laps_complete = st.number_input("Laps Completed", min_value=1, max_value=500, value=90)
pos_start     = st.number_input("Starting Position", min_value=1, max_value=30, value=3)
status        = st.selectbox("Status (0 = running, 1 = DNF/other)", [0, 1], index=0)
track         = st.selectbox("Track", TRACK_OPTIONS)
track_type    = st.selectbox("Track Type", TRACKTYPE_OPTIONS)
team          = st.selectbox("Team", TEAM_OPTIONS)
driver        = st.text_input("Driver", DRIVER_OPTIONS[0])

if st.button("Predict"):
    try:
        # 1) Zero-out all features
        features = dict.fromkeys(FEATURE_COLS, 0)

        # 2) Fill numeric fields
        features['BestLapTimeSec'] = time_to_seconds(best_lap)
        features['LapsComplete']   = laps_complete
        features['PositionStart']  = pos_start
        features['Status']         = status

        # 3) One-hot for known categories
        features[track]      = 1
        features[track_type] = 1
        features[team]       = 1

        # 4) Driver: if in your training list, flip its dummy; otherwise leave all-zero (baseline)
        if driver in DRIVER_OPTIONS:
            features[driver] = 1
        else:
            st.warning(f"‘{driver}’ not in training set; using baseline driver.")

        input_df = pd.DataFrame([features], columns=FEATURE_COLS)
        pred = model.predict(input_df)[0]
        st.success(f"Predicted Finishing Position: {pred:.2f}")

    except Exception as e:
        st.error(f"Error in prediction: {e}")

# Instructions
st.sidebar.header("Instructions")
st.sidebar.markdown(
    "1. Ensure `indycar_xgb.json` and the dataset CSV are in this folder.  \n"
    "2. Run with: `streamlit run streamlit_indycar_predictor.py`.  \n"
    "3. Enter values and hit **Predict**."
)