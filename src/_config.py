from typing import Final

# Figure directory
IMG_DIR: Final = "img"

# Trial hyperparameters
N_OPPONENTS: Final = 2  # Number of virtual opponents
N_TRIALS: Final = 15  # Number of trials against each opponent
MAX_LATENCY: Final = 2

# Extrema for opponent belief questionnaires
MIN_BELIEF_COHORT_B: Final = 0
MAX_BELIEF_COHORT_B: Final = 5
MIN_BELIEF_COHORT_A: Final = 0
MAX_BELIEF_COHORT_A: Final = 10

# Outlier threshold for number of shocks + belief
MIN_BELIEF: Final = 2.5
MIN_SHOCKS: Final = 3
MAX_SHOCKS: Final = 27

# Color palettes
QUALIT_PALETTE: Final = "colorblind"  # Default qualitative palette

# P-value thresholds for statannoations
PVALUE_MAP: Final = [[1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]

# Random seed for reproducibility
RANDOM_SEED: Final = 42

# Clustering parameters
K_MIN: Final = 2
K_MAX: Final = 13
DEFAULT_CLUSTERING_FEATURES = [
    "Kp",
    "Kr1",
    "Krc",
    "Kwc",
    "R2",
    "shock_opp1",
    "shock_opp2",
    "first_shock",
    "belief_opp1",
    "belief_opp2",
]

# Clusters
# In _config.py or at the top of fig3:
CLUSTERS = [
    {"label": 2, "name": "Non-aggressive"},
    {"label": 1, "name": "Proactive"},
    {"label": 0, "name": "Reactive"},
]

PALETTE = {0: "#de8f05", 1: "#029e73", 2: "#0173b2"}

CLUSTER_NAMES = {cl["label"]: cl["name"] for cl in CLUSTERS}
