"""Category labels and thresholds shared by build_dataset.py, build_deploy_data.py (which computes the aggregates) and app.py."""

FLOOD_ORDER = ["X / B / C (moderate-minimal)", "A / AH / AO / AR / A99 (no base elevation, shallow, special)",
               "AE / A1-A30 (base elevation known)", "V / VE (velocity, coastal)", "D / unknown"]
OCC_ORDER = ["Single-family", "2-4 units", "5+ units", "Mobile home", "Condo association",
             "Unit in multi-unit building", "Non-residential", "Unknown"]
AGE_LABELS = ["0-10 yrs", "11-20", "21-30", "31-40", "41-50", "51-60", "60+"]
AGE_BINS = [-1, 10, 20, 30, 40, 50, 60, 500]
ELEVATED_LABELS = ["Not elevated", "Elevated"]

# FEMA's `floodEvent` label groups. "Named storm" labels start with Hurricane or Tropical Storm.
EVENT_GROUPS = ["Named storm", "Other FEMA-labeled event", "No event label"]
NAMED_STORM_PATTERN = r"^(?:Hurricane|Tropical Storm)\b"

# FEMA labels two events with one name or one event with two names; these are the only such cases in the data.
# Alberto (1994 and 2006) is shown with its year; Georges (1998) was split by FEMA into a Keys and a Panhandle label.
EVENT_MERGES = {"Hurricane Georges (Keys)": "Hurricane Georges", "Hurricane Georges (Panhandle)": "Hurricane Georges"}
EVENT_NEEDS_YEAR = {"Tropical Storm Alberto"}

# Date of the OpenFEMA pull behind the published numbers (see the SHA-256 in docs/METHODOLOGY.md).
FEMA_PULL_DATE = "2026-10-20"

# Groups with fewer claims than this are flagged or left out of rankings and medians.
MIN_CLAIMS = 30

# Building coverage: the policy's building limit. 40% of claims sit at exactly $250,000 and 2.6% have no building coverage,
# so those get bands of their own.
COVERAGE_BINS = [-1, 0, 50000, 100000, 150000, 249999, 250000, 1e12]
COVERAGE_LABELS = ["No building coverage", "Up to $50k", "$50k-$100k", "$100k-$150k", "$150k-$250k", "$250k", "Over $250k"]

# FEMA's numberOfFloorsInTheInsuredBuilding is a code, not a count (FEMA data dictionary).
FLOOR_LABELS = {1: "One floor", 2: "Two floors", 3: "Three or more floors", 4: "Split-level",
                5: "Manufactured home or trailer on foundation", 6: "Townhouse, three or more floors"}
FLOOR_ORDER = list(FLOOR_LABELS.values()) + ["Not recorded"]
PRIMARY_LABELS = ["Primary residence", "Not a primary residence"]
FIRM_LABELS = ["Built before the first flood map", "Built after the first flood map"]

# factor name -> (column in the claims table, display order); this is the order of the Risk Factors dropdown
FACTOR_COLUMNS = {
    "Flood zone": ("flood_zone_group", FLOOD_ORDER),
    "Occupancy type": ("occupancy_group", OCC_ORDER),
    "Building floors": ("floors_group", FLOOR_ORDER),
    "Elevated building": ("elevated", ELEVATED_LABELS),
    "Building age": ("age_band", AGE_LABELS),
    "Built after first flood map": ("firm_group", FIRM_LABELS),
    "Primary residence": ("primary_residence", PRIMARY_LABELS),
    "Building coverage": ("coverage_band", COVERAGE_LABELS),
}
