import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
import joblib

# --- Helper Functions ---
# Convert 'BestLapTime' to seconds
def time_to_seconds(time_str):
    if isinstance(time_str, str):
        parts = time_str.split(':')
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        elif len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return np.nan

def get_track_meta(track_code: str):
    meta = TRACKS_MAP.get(track_code, {})
    laps = meta.get('laps', np.nan)
    iD = meta.get('id', None)
    return laps, iD



# --- Load Model & Schema ---
@st.cache_resource
def load_model_and_schema(model_path: str, data_path: str):
    
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    df['BestLapTime_seconds'] = df['BestLapTime'].apply(time_to_seconds)
    X_cols = df.drop(columns=['BestLapTime', 'PositionFinish']).columns.tolist()
    return model, X_cols

model, FEATURE_COLS = load_model_and_schema(
    model_path="C:\\Users\\ripa_\\Desktop\\Programing\\IndyCar_Project\\models\\indycar_cat_model_v1.pkl",
    data_path="C:\\Users\\ripa_\\Desktop\\Programing\\IndyCar_Project\\datasets\\IndyCar_Regression_Data_vO2.csv"
)

# --- Static Lists ---
TRACKS_MAP = {
    'BAR' : {'laps': 90, 'id': 1},
    'DET' : {'laps': 100, 'id': 2},
    'GPI' : {'laps': 85, 'id': 3},
    'IND' : {'laps': 200, 'id': 4},
    'IOW' : {'laps': 275, 'id': 5},
    'LB' : {'laps': 90, 'id': 6},
    'MIL' : {'laps': 250, 'id': 7},
    'MOH' : {'laps': 90, 'id': 8},
    'NAS' : {'laps': 206, 'id': 9},
    'NASH' : {'laps': 80, 'id': 10},
    'POR' : {'laps': 110, 'id': 11},
    'R-AM' : {'laps': 55, 'id': 12},
    'STL' : {'laps': 260, 'id': 13},
    'STP' : {'laps': 100, 'id': 14},
    'TEX' : {'laps': 250, 'id': 15},
    'THRM' : {'laps': 65, 'id': 16},
    'TOR' : {'laps': 90, 'id': 17},
    'WRLS' : {'laps': 95, 'id': 18}
}

TEAMS_MAP  = {
    'A.J. Foyt Enterprises' : 1,
    'Andretti Global' : 2,
    'Arrow McLaren' : 3,
    'Chip Ganassi Racing' : 4,
    'Dale Coyne Racing' : 5,
    'Ed Carpenter Racing' : 6,
    'Juncos Hollinger Racing' : 7,
    'Meyer Shank Racing' : 8,
    'PREMA Racing' : 9,
    'Rahal Letterman Lanigan Racing' : 10,
    'Team Penske' : 11
}

DRIVERS_MAP = {
    'Alex Palou' : 4931,
    'Alexander Rossi' : 4587,
    'Callum Ilott' : 4939,
    'Christian Lundgaard' : 4938,
    'Christian Rasmussen' : 4920,
    'Colton Herta' : 4511,
    'Conor Daly' : 4218,
    'David Malukas' : 4636,
    'Devlin DeFrancesco' : 4940,
    'Felix Rosenqvist' : 4588,
    'Graham Rahal' : 3668,
    'Jacob Abel' : 4644,
    'Josef Newgarden' : 4215,
    'Kyffin Simpson' : 4944,
    'Kyle Kirkwood' : 4623,
    'Louis Foster' : 4949,
    'Marcus Armstrong' : 4948,
    'Marcus Ericsson' : 4905,
    'Nolan Siegel' : 4915,
    "Pato O'Ward" : 4559,
    'Robert Shwartzman' : 4918,
    'Rinus VeeKay' : 4614,
    'Santino Ferrucci' : 4897,
    'Scott Dixon' : 3628,
    'Scott McLaughlin' : 4932,
    'Sting Ray Robb' : 4613,
    'Will Power' : 3667
}

TRACK_TYPE_DISPLAY = {
    "Road Course": 1,
    "Oval":       0,
    "Street Circuit": 2
}



# Filter only those that match model's feature columns
TRACK_OPTIONS  = [t for t in TRACKS_MAP if t in FEATURE_COLS or 'TrackID' in FEATURE_COLS]
TEAM_OPTIONS   = [t for t in TEAMS_MAP if (('TeamID' in FEATURE_COLS) or (t in FEATURE_COLS))]
DRIVER_OPTIONS = [d for d in DRIVERS_MAP if (('DriverID' in FEATURE_COLS) or (d in FEATURE_COLS))]
TRACK_TYPE_OPTIONS = list(TRACK_TYPE_DISPLAY.keys())

# --- Streamlit UI ---
st.title("IndyCar Race Result Predictor")
st.markdown("Fill in the fields below and click **Predict** to estimate finishing position.")


# Inputs
best_lap      = st.text_input("Best Qualy Lap Time (MM:SS.s)", "01:08.2")
pos_start     = st.number_input("Starting Position", min_value=1, max_value=30, value=3)
track         = st.selectbox("Track", TRACK_OPTIONS)
track_type    = st.selectbox("Track Type", TRACK_TYPE_OPTIONS)
team          = st.selectbox("Team", TEAM_OPTIONS)
driver        = st.selectbox("Driver", DRIVER_OPTIONS)

selected_track_code = track
laps_complete, track_id = get_track_meta(selected_track_code)

if st.button("Predict"):
    try:
        # Initialize feature vector with zeros
        features = dict.fromkeys(FEATURE_COLS, 0)

        bl_seconds = time_to_seconds(best_lap)
        features['BestLapTime_seconds'] = float(bl_seconds)
        features['LapsComplete']   = laps_complete
        features['PositionStart']  = pos_start
        features['Status']         = 0

        if 'TrackID' in FEATURE_COLS and track_id is not None:
            features['TrackID'] = track_id
        if selected_track_code in FEATURE_COLS:
            features[selected_track_code] = 1

        #(0 = Oval, 1 = Road, 2 = Street)
        if 'TrackType' in FEATURE_COLS and track_type in TRACK_TYPE_DISPLAY:
            features['TrackType'] =  TRACK_TYPE_DISPLAY[track_type]

        if 'TeamID' in FEATURE_COLS:
            features['TeamID'] = int(TEAMS_MAP.get(team, 0))

        if 'DriverID' in FEATURE_COLS:
            features['DriverID'] = int(DRIVERS_MAP.get(driver, 0))

        # Create DataFrame & predict
        input_df = pd.DataFrame([features], columns=FEATURE_COLS)
        pred = model.predict(input_df)[0]
        st.success(f"Predicted Finishing Position: {pred:.2f}")

    except Exception as e:
        st.error(f"Error in prediction: {e}")