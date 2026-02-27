import streamlit as st
import pandas as pd
import numpy as np
from pycaret.regression import load_model, predict_model
from pathlib import Path

Base_dir = Path(__file__).resolve().parent

# --- Model Paths ---
PRE_QUALI_MODEL_PATH  = Base_dir / "models" / "indycar_cat_lgbm_prequaly_model_v1"
POST_QUALI_MODEL_PATH = Base_dir / "models" / "indycar_postqualy_model_v1"
DATA_PATH             = Base_dir / "datasets" / "IndyCar_dataset_v12.csv"

# --- Load model on demand ---
@st.cache_resource
def load_selected_model(model_path: str):
    return load_model(model_path)

@st.cache_data
def load_feature_cols(data_path: str, is_post_quali: bool):
    df = pd.read_csv(data_path)
    drop_cols = [
        "DriverName", "TeamName", "CarEngine", "EventName", "Track", "EventTrackType",
        "EventDate", "EventDateFormatted", "EventID", "Era",
        "Status", "StatusID", "PositionFinish"
    ]
    if not is_post_quali:
        drop_cols.append("PositionStart")
    return df.drop(columns=drop_cols, errors='ignore').columns.tolist()

@st.cache_data
def load_dataset_means(data_path: str):
    df = pd.read_csv(data_path)
    return df.select_dtypes(include=[np.number]).mean().to_dict()

@st.cache_data
def load_latest_driver_stats(data_path: str):
    df = pd.read_csv(data_path)
    df["EventDate"] = pd.to_datetime(df["EventDate"])
    # Get most recent row per driver
    return df.sort_values("EventDate").groupby("DriverID").last().to_dict("index")

@st.cache_data
def load_latest_team_stats(data_path: str):
    df = pd.read_csv(data_path)
    df["EventDate"] = pd.to_datetime(df["EventDate"])
    # Get most recent row per team
    return df.sort_values("EventDate").groupby("TeamID").last().to_dict("index")

# --- Driver Map: Name -> {id, team_id, team_name, rookie} ---
DRIVERS_MAP = {
    "Marcus Armstrong":    {"id": 4948, "team_id": 19, "team_name": "Meyer Shank Racing",              "rookie": False},
    "Caio Collet":         {"id": None, "team_id": 1,  "team_name": "A.J. Foyt Enterprises",           "rookie": True},
    "Scott Dixon":         {"id": 3628, "team_id": 9,  "team_name": "Chip Ganassi Racing",             "rookie": False},
    "Marcus Ericsson":     {"id": 4905, "team_id": 3,  "team_name": "Andretti Global",                 "rookie": False},
    "Santino Ferrucci":    {"id": 4897, "team_id": 1,  "team_name": "A.J. Foyt Enterprises",           "rookie": False},
    "Louis Foster":        {"id": 4949, "team_id": 24, "team_name": "Rahal Letterman Lanigan Racing",  "rookie": False},
    "Romain Grosjean":     {"id": 4935, "team_id": 11, "team_name": "Dale Coyne Racing",               "rookie": False},
    "Dennis Hauger":       {"id": None, "team_id": 11, "team_name": "Dale Coyne Racing",               "rookie": True},
    "Kyle Kirkwood":       {"id": 4623, "team_id": 3,  "team_name": "Andretti Global",                 "rookie": False},
    "Christian Lundgaard": {"id": 4938, "team_id": 4,  "team_name": "Arrow McLaren",                   "rookie": False},
    "David Malukas":       {"id": 4636, "team_id": 28, "team_name": "Team Penske",                     "rookie": False},
    "Scott McLaughlin":    {"id": 4932, "team_id": 28, "team_name": "Team Penske",                     "rookie": False},
    "Josef Newgarden":     {"id": 4215, "team_id": 28, "team_name": "Team Penske",                     "rookie": False},
    "Pato O'Ward":         {"id": 4559, "team_id": 4,  "team_name": "Arrow McLaren",                   "rookie": False},
    "Alex Palou":          {"id": 4931, "team_id": 9,  "team_name": "Chip Ganassi Racing",             "rookie": False},
    "Will Power":          {"id": 3667, "team_id": 3,  "team_name": "Andretti Global",                 "rookie": False},
    "Graham Rahal":        {"id": 3668, "team_id": 24, "team_name": "Rahal Letterman Lanigan Racing",  "rookie": False},
    "Christian Rasmussen": {"id": 4920, "team_id": 15, "team_name": "Ed Carpenter Racing",             "rookie": False},
    "Sting Ray Robb":      {"id": 4613, "team_id": 16, "team_name": "Juncos Hollinger Racing",         "rookie": False},
    "Felix Rosenqvist":    {"id": 4588, "team_id": 19, "team_name": "Meyer Shank Racing",              "rookie": False},
    "Alexander Rossi":     {"id": 4587, "team_id": 15, "team_name": "Ed Carpenter Racing",             "rookie": False},
    "Mick Schumacher":     {"id": None, "team_id": 24, "team_name": "Rahal Letterman Lanigan Racing",  "rookie": True},
    "Nolan Siegel":        {"id": 4915, "team_id": 4,  "team_name": "Arrow McLaren",                   "rookie": False},
    "Kyffin Simpson":      {"id": 4944, "team_id": 9,  "team_name": "Chip Ganassi Racing",             "rookie": False},
    "Rinus VeeKay":        {"id": 4614, "team_id": 16, "team_name": "Juncos Hollinger Racing",         "rookie": False},
}

# --- Team Map: TeamID -> {engine, engine_id} ---
TEAMS_MAP = {
    1:  {"name": "A.J. Foyt Enterprises",          "engine": "Chevrolet", "engine_id": 0},
    3:  {"name": "Andretti Global",                 "engine": "Honda",     "engine_id": 1},
    4:  {"name": "Arrow McLaren",                   "engine": "Chevrolet", "engine_id": 0},
    9:  {"name": "Chip Ganassi Racing",             "engine": "Honda",     "engine_id": 1},
    11: {"name": "Dale Coyne Racing",               "engine": "Honda",     "engine_id": 1},
    15: {"name": "Ed Carpenter Racing",             "engine": "Chevrolet", "engine_id": 0},
    16: {"name": "Juncos Hollinger Racing",         "engine": "Chevrolet", "engine_id": 0},
    19: {"name": "Meyer Shank Racing",              "engine": "Honda",     "engine_id": 1},
    24: {"name": "Rahal Letterman Lanigan Racing",  "engine": "Honda",     "engine_id": 1},
    28: {"name": "Team Penske",                     "engine": "Chevrolet", "engine_id": 0},
}

# --- Track Map: Name -> {id, type, type_id, is_new} ---
TRACKS_MAP = {
    "Streets of St. Petersburg":      {"id": 21, "type": "Street", "type_id": 2, "is_new": False},
    "Phoenix Raceway":                {"id": 10, "type": "Oval",   "type_id": 0, "is_new": False},
    "Streets of Arlington":           {"id": None,"type": "Street", "type_id": 2, "is_new": True},
    "Barber Motorsports Park":        {"id": 1,  "type": "Road",   "type_id": 1, "is_new": False},
    "Streets of Long Beach":          {"id": 18, "type": "Street", "type_id": 2, "is_new": False},
    "Indianapolis Motor Speedway (Road)": {"id": 3,  "type": "Road",   "type_id": 1, "is_new": False},
    "Indianapolis Motor Speedway (Oval)": {"id": 4,  "type": "Oval",   "type_id": 0, "is_new": False},
    "Streets of Detroit":             {"id": 17, "type": "Street", "type_id": 2, "is_new": False},
    "World Wide Technology Raceway":  {"id": 28, "type": "Oval",   "type_id": 0, "is_new": False},
    "Road America":                   {"id": 14, "type": "Road",   "type_id": 1, "is_new": False},
    "Mid-Ohio Sports Car Course":     {"id": 6,  "type": "Road",   "type_id": 1, "is_new": False},
    "Nashville Superspeedway":        {"id": 8,  "type": "Oval",   "type_id": 0, "is_new": False},
    "Portland International Raceway": {"id": 12, "type": "Road",   "type_id": 1, "is_new": False},
    "Streets of Markham":             {"id": None,"type": "Street", "type_id": 2, "is_new": True},
    "Streets of Washington":          {"id": None,"type": "Street", "type_id": 2, "is_new": True},
    "WeatherTech Raceway Laguna Seca":{"id": 27, "type": "Road",   "type_id": 1, "is_new": False},
    "Milwaukee Mile":                 {"id": 7,  "type": "Oval",   "type_id": 0, "is_new": False},
}

# --- Helper: denormalize position ---
def denormalize_position(norm_val: float, field_size: int = 25) -> int:
    raw = norm_val * (field_size - 1) + 1
    return int(round(np.clip(raw, 1, field_size)))

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="IndyCar Predictor", layout="centered")
st.title("IndyCar Race Result Predictor")
st.markdown("Select your inputs below and click **Predict** to estimate a driver's finishing position.")

# --- Model Mode Selection ---
st.subheader("Prediction Mode")
mode = st.radio(
    "Select prediction stage:",
    ["Pre-Qualy", "Post-Qualy"],
    horizontal=True
)
is_post_quali = mode.startswith("Post")

# --- Load correct model and features ---
model_path = POST_QUALI_MODEL_PATH if is_post_quali else PRE_QUALI_MODEL_PATH
model = load_selected_model(str(model_path))
FEATURE_COLS = load_feature_cols(str(DATA_PATH), is_post_quali)
DATASET_MEANS = load_dataset_means(str(DATA_PATH))
DRIVER_STATS = load_latest_driver_stats(str(DATA_PATH))
TEAM_STATS = load_latest_team_stats(str(DATA_PATH))

st.divider()

# --- Inputs ---
st.subheader("Race Setup")

col1, col2 = st.columns(2)

with col1:
    driver_name = st.selectbox("Driver", list(DRIVERS_MAP.keys()))

with col2:
    track_name = st.selectbox("Track", list(TRACKS_MAP.keys()))

# Post-quali only: show starting position input
pos_start = None
if is_post_quali:
    pos_start = st.number_input("Starting Position", min_value=1, max_value=33, value=10)

# --- Auto-resolve driver/team/track info ---
driver_info = DRIVERS_MAP[driver_name]
team_info   = TEAMS_MAP[driver_info["team_id"]]
track_info  = TRACKS_MAP[track_name]


# --- Warnings for limited data ---
warnings = []
if driver_info["rookie"] or driver_info["id"] is None:
    warnings.append(f"**{driver_name}** is a rookie or new to the series — prediction will be based on team stats and track type only.")
if track_info["is_new"]:
    warnings.append(f"**{track_name}** is a new track with no historical data — prediction will be based on track type only.")

for w in warnings:
    st.warning(w)

st.divider()

# --- Predict Button ---
if st.button("Predict Finishing Position", use_container_width=True):
    try:
        # Start with dataset means as baseline for all features
        features = {col: DATASET_MEANS.get(col, 0) for col in FEATURE_COLS}

        # Driver features
        features["Rookie"] = int(driver_info["rookie"] or driver_info["id"] is None)

        if driver_info["id"] and not driver_info["rookie"]:
            # Known driver — pull their latest real stats
            driver_row = DRIVER_STATS.get(driver_info["id"], {})
            for col in ["DRFAvg", "DTAvg", "DTTAvg", "DNFRate", "TDNFRate",
                        "DriverElo", "DriverTElo", "DriverRitmo"]:
                if col in FEATURE_COLS and col in driver_row:
                    features[col] = driver_row[col]
            if "DriverID" in FEATURE_COLS:
                features["DriverID"] = int(driver_info["id"])
        else:
            # Rookie/new driver — use team stats as proxy
            if "DriverID" in FEATURE_COLS:
                features["DriverID"] = int(DATASET_MEANS.get("DriverID", 0))
            for col in ["DRFAvg", "DTAvg", "DTTAvg", "DNFRate", "TDNFRate",
                        "DriverElo", "DriverTElo", "DriverRitmo"]:
                if col in FEATURE_COLS:
                    features[col] = DATASET_MEANS.get(col, 0)

        # Team features
        team_row = TEAM_STATS.get(driver_info["team_id"], {})
        for col in ["TeamElo", "TeamTElo", "TeamRitmo", "TeamDNFRate"]:
            if col in FEATURE_COLS and col in team_row:
                features[col] = team_row[col]
        if "TeamID" in FEATURE_COLS:
            features["TeamID"] = int(driver_info["team_id"])

        # Track features
        if "TrackID" in FEATURE_COLS:
            features["TrackID"] = int(track_info["id"]) if track_info["id"] else int(DATASET_MEANS.get("TrackID", 0))
        if "EventTrackTypeID" in FEATURE_COLS:
            features["EventTrackTypeID"] = int(track_info["type_id"])

        # If new track — zero out track-specific stats
        if track_info["is_new"]:
            for col in ["TRP", "TTP", "TeamRitmo"]:
                if col in FEATURE_COLS:
                    features[col] = DATASET_MEANS.get(col, 0)

        # Post-quali: add starting position
        if is_post_quali and pos_start is not None:
            if "PositionStart" in FEATURE_COLS:
                features["PositionStart"] = int(pos_start)

        # Build input DataFrame and predict
        input_df = pd.DataFrame([features], columns=FEATURE_COLS)
        result = predict_model(model, data=input_df)
        norm_pred = result["prediction_label"].iloc[0]

        # Denormalize
        finish_pos = denormalize_position(norm_pred)

        st.success(f"### Predicted Finishing Position: P{finish_pos}")
        st.caption(f"Raw Position: {norm_pred * (25 - 1) + 1:.2f}")

        if warnings:
            st.info("Note: This prediction has lower confidence due to limited historical data for this driver/track combination.")

    except Exception as e:
        st.error(f"Prediction error: {e}")