"""Category labels and thresholds shared by pipeline/build_dataset.py, pipeline/build_deploy_data.py (which computes the aggregates) and app.py."""

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
FEMA_PULL_DATE = "2026-11-03"

# Groups with fewer claims than this are flagged or left out of rankings and medians.
MIN_CLAIMS = 30

# The three storms whose claims carry the 2020s; summary.json shows the 2020s median without them.
BIG_THREE_STORMS = ("Hurricane Ian", "Hurricane Helene", "Hurricane Milton")

# FEMA's two fields that code why a claim was closed without payment (building, else contents), in the groups the report uses.
NON_PAYMENT_FIELDS = ("nonPaymentReasonBuilding", "nonPaymentReasonContents")
NON_PAYMENT_REASON_GROUPS = {"Below the deductible": ("01",), "No damage or not a flood": ("06", "20"), "Administrative error": ("98", "99")}
OTHER_REASON, NO_REASON = "Other reason", "No reason recorded"
NON_PAYMENT_GROUP_ORDER = [*NON_PAYMENT_REASON_GROUPS, OTHER_REASON, NO_REASON]

# Intervals for a median resample whole storms (claims from one storm are not independent of each other).
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20261110

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

# Labels for FEMA's coded fields, from the OpenFEMA data dictionary for NfipClaims. A code the dictionary does not list is "Other or
# not recorded" (cause) or "Not recorded" (the rest).
CAUSE_LABELS = {"4": "Accumulation of rainfall or snowmelt", "1": "Tidal water overflow", "2": "Stream, river or lake overflow",
                "3": "Alluvial fan overflow", "0": "Other causes"}
CAUSE_ORDER = list(CAUSE_LABELS.values()) + ["Other or not recorded"]

# Water depth is in inches (FEMA notes some entries were recorded in feet); negative values count as not recorded.
DEPTH_BINS = [-1, 0, 6, 12, 36, 1e9]
DEPTH_LABELS = ["0 in", "1-6 in", "7-12 in", "13-36 in", "Over 36 in"]
DEPTH_ORDER = DEPTH_LABELS + ["Not recorded"]

# Feet between the building's lowest floor (as used for rating) and the base flood elevation.
ELEVATION_LABELS = ["Below the base flood level", "At the base flood level", "1 ft above", "2 ft above", "3 ft or more above"]
ELEVATION_ORDER = ELEVATION_LABELS + ["Not recorded"]

DEDUCTIBLE_LABELS = {"H": "$200", "0": "$500", "9": "$750", "1": "$1,000", "F": "$1,250", "G": "$1,500", "2": "$2,000", "3": "$3,000",
                     "4": "$4,000", "5": "$5,000", "A": "$10,000", "B": "$15,000", "C": "$20,000", "D": "$25,000", "E": "$50,000"}
DEDUCTIBLE_ORDER = list(DEDUCTIBLE_LABELS.values()) + ["Not recorded"]

# Code 0 is not in FEMA's list of foundation types; it is 95-100% of claims before 2010, so it is read as "not recorded".
FOUNDATION_LABELS = {"1": "Slab", "2": "Basement", "3": "Crawlspace", "4": "Elevated, no enclosure (post, pile or pier)",
                     "5": "Elevated with enclosure (post, pile or pier)", "6": "Elevated with enclosure (not post, pile or pier)"}
FOUNDATION_ORDER = list(FOUNDATION_LABELS.values()) + ["Not recorded"]

# factor name -> (column in the claims table, display order); the Factors dropdown lists them alphabetically
_FACTORS = {
    "Flood zone": ("flood_zone_group", FLOOD_ORDER),
    "Occupancy type": ("occupancy_group", OCC_ORDER),
    "Building floors": ("floors_group", FLOOR_ORDER),
    "Foundation type": ("foundation_group", FOUNDATION_ORDER),
    "Elevated building": ("elevated", ELEVATED_LABELS),
    "Elevation vs base flood level": ("elevation_band", ELEVATION_ORDER),
    "Building age": ("age_band", AGE_LABELS),
    "Built after first flood map": ("firm_group", FIRM_LABELS),
    "Primary residence": ("primary_residence", PRIMARY_LABELS),
    "Building coverage": ("coverage_band", COVERAGE_LABELS),
    "Building deductible": ("deductible_group", DEDUCTIBLE_ORDER),
    "Cause of flooding": ("cause_group", CAUSE_ORDER),
    "Flood water depth": ("depth_band", DEPTH_ORDER),
}
FACTOR_COLUMNS = dict(sorted(_FACTORS.items()))
