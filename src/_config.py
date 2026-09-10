from typing import Final

# Figure directory
DEFAULT_DATA_DIR: Final = "data_v2"
DEFAULT_BRMS_DIR: Final = f"{DEFAULT_DATA_DIR}/brms"
DEFAULT_CLUSTER_DIR_A: Final = f"{DEFAULT_DATA_DIR}/cohort_a/clustering"
DEFAULT_CLUSTER_DIR_B: Final = f"{DEFAULT_DATA_DIR}/cohort_b/clustering"
DEFAULT_PHYSIO_DIR_A: Final = f"{DEFAULT_DATA_DIR}/cohort_a/physio"
DEFAULT_PHYSIO_DIR_B: Final = f"{DEFAULT_DATA_DIR}/cohort_b/physio"
DEFAULT_SHARED_DIR: Final = f"{DEFAULT_DATA_DIR}/shared"

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
FEATURE_LABELS = {
    "Kp": r"$K_p$",
    "Kr1": r"$K_{r_1}$",
    "Krc": r"$K_{r_c}$",
    "Kwc": r"$K_{w_c}$",
    "R2": r"$R^2$",
    "shock_opp1": "Shocks\n(Opp. 1)",
    "shock_opp2": "Shocks\n(Opp. 2)",
    "first_shock": "1st shock",
}
DEFAULT_CLUSTERING_FEATURES = list(FEATURE_LABELS.keys())

# Clusters
CLUSTERS = [
    {"label": 1, "name": "Non-aggressive", "annot_xy": (-3, -3.5)},
    {"label": 2, "name": "Proactive", "annot_xy": (2, -3.25)},
    {"label": 0, "name": "Reactive", "annot_xy": (-1.5, 4)},
]
PALETTE = {0: "#de8f05", 1: "#0173b2", 2: "#029e73"}
CLUSTER_NAMES = {cl["label"]: cl["name"] for cl in CLUSTERS}
CLUSTER_PALETTE = {cl["name"]: PALETTE[cl["label"]] for cl in CLUSTERS}
