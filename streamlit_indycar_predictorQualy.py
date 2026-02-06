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
    X_cols = df.drop(columns=['BestLapTime', 'PositionStart']).columns.tolist()
    return model, X_cols

model, FEATURE_COLS = load_model_and_schema(
    model_path="C:\\Users\\ripa_\\Downloads\\indycar_xgb_v3.json",
    data_path='C:\\Users\\ripa_\\Downloads\\IndyCar Regression Data_v2.csv'
)

# --- Static Lists ---
TRACKS = ['BAR','BEL','GPI','IND','IOW','LB', 'MIL','MOH','NAS', 'NASH', 'POR', 'R-AM', 'STL', 'STP', 'TEX', 'THRM', 'TOR', 'WRLS']
TEAMS  = [
    'A.J. Foyt Enterprises', 'Andretti Global', 'Arrow McLaren','Chip Ganassi Racing','Dale Coyne Racing','Ed Carpenter Racing',
    'Juncos Hollinger Racing','Meyer Shank Racing','PREMA Racing', 'Rahal Letterman Lanigan Racing','Team Penske'
]
DRIVERS = [
    'Alex Palou','Alexander Rossi','Callum Ilott','Christian Lundgaard','Christian Rasmussen','Colton Herta','Conor Daly',
    'David Malukas','Devlin DeFrancesco','Felix Rosenqvist','Graham Rahal','Jacob Abel','Josef Newgarden',
    'Kyffin Simpson','Kyle Kirkwood','Louis Foster', 'Marcus Armstrong','Marcus Ericsson','Nolan Siegel',
    "Pato O'Ward", 'Robert Shwartzman', 'Rinus VeeKay','Santino Ferrucci','Scott Dixon','Scott McLaughlin','Sting Ray Robb'
]

TRACK_TYPE_DISPLAY = {
    "Road Course": "R",
    "Oval":        "O",
    "Street Circuit": None
}

# Filter only those that match model's feature columns
TRACK_OPTIONS      = [t for t in TRACKS  if t in FEATURE_COLS]
TEAM_OPTIONS       = [t for t in TEAMS   if t in FEATURE_COLS]
DRIVER_OPTIONS    = [d for d in DRIVERS if d in FEATURE_COLS]
TRACK_TYPE_OPTIONS = list(TRACK_TYPE_DISPLAY.keys())

# --- Streamlit UI ---
st.title("IndyCar Qualy Predictor")
st.markdown("Fill in the fields below and click **Predict** to estimate finishing position.")

# Inputs
best_lap      = st.text_input("Best Lap Time (MM:SS.s)", "01:08.2")
laps_complete = st.number_input("Laps Completed", min_value=1, max_value=500, value=90)
pos_fin     = st.number_input("Last Race Result", min_value=1, max_value=27, value=3)
status        = st.selectbox("Status (0 = running, 1 = DNF/other)", [0, 1], index=0)
track         = st.selectbox("Track", TRACK_OPTIONS)
track_type    = st.selectbox("Track Type", TRACK_TYPE_OPTIONS)
team          = st.selectbox("Team", TEAM_OPTIONS)
driver        = st.text_input("Driver", DRIVER_OPTIONS[0] if DRIVER_OPTIONS else "")

if st.button("Predict"):
    try:
        # Initialize feature vector with zeros
        features = dict.fromkeys(FEATURE_COLS, 0)

        # Numeric features
        features['BestLapTimeSec'] = time_to_seconds(best_lap)
        features['LapsComplete']   = laps_complete
        features['PositionFinish']  = pos_fin
        features['Status']         = status

        # One-hot for track
        features[track] = 1

        # One-hot for track type: reset both, then set selected if not baseline
        for col in ['R', 'O']:
            if col in FEATURE_COLS:
                features[col] = 0
        selected_col = TRACK_TYPE_DISPLAY[track_type]
        if selected_col:
            features[selected_col] = 1

        # One-hot for team
        features[team] = 1

        # One-hot for driver if known; otherwise baseline
        if driver in DRIVER_OPTIONS:
            features[driver] = 1
        else:
            st.warning(f"‘{driver}’ not in training set; using baseline driver.")

        # Create DataFrame & predict
        input_df = pd.DataFrame([features], columns=FEATURE_COLS)
        pred = model.predict(input_df)[0]
        st.success(f"Predicted Qualy Position: {pred:.2f}")

    except Exception as e:
        st.error(f"Error in prediction: {e}")

# Instructions
st.sidebar.header("Instructions")
st.sidebar.markdown(
    "1. Ensure `indycar_xgb.json` and the dataset CSV are in this folder.  \n"
    "2. Run with: `streamlit run streamlit_indycar_predictor.py`.  \n"
    "3. Enter values and hit **Predict**."
)


