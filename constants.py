"""Category labels shared by build_deploy_data.py (which computes the aggregates) and app.py (which displays them)."""

STORM_ORDER = ["No storm", "Tropical Depression", "Tropical Storm", "Category 1", "Category 2", "Category 3",
               "Category 4", "Category 5"]
FLOOD_ORDER = ["X / B / C (moderate-minimal)", "A / AH / AO / AR / A99 (no base elevation, shallow, special)",
               "AE / A1-A30 (base elevation known)", "V / VE (velocity, coastal)", "D / unknown"]
OCC_ORDER = ["Single-family", "2-4 units", "5+ units", "Mobile home", "Condo association",
             "Unit in multi-unit building", "Non-residential", "Unknown"]
AGE_LABELS = ["0-10 yrs", "11-20", "21-30", "31-40", "41-50", "51-60", "60+"]
AGE_BINS = [-1, 10, 20, 30, 40, 50, 60, 500]
DIST_LABELS = ["No storm match", "0-25 mi", "25-50 mi", "50-100 mi", "100-150 mi"]
ELEVATED_LABELS = ["Not elevated", "Elevated"]

# Saffir-Simpson wind thresholds in KNOTS (HURDAT2 reports maximum sustained wind in knots): tropical depression below 34,
# tropical storm 34-63, Category 1 64-82, Category 2 83-95, Category 3 96-112, Category 4 113-136, Category 5 137 and above.
WIND_BREAKS_KT = (34, 64, 83, 96, 113, 137)

# factor name -> (column in the claims table, display order)
FACTOR_COLUMNS = {
    "Storm category": ("storm_category", STORM_ORDER),
    "Flood zone": ("flood_zone_group", FLOOD_ORDER),
    "Occupancy type": ("occupancy_group", OCC_ORDER),
    "Elevated building": ("elevated", ELEVATED_LABELS),
    "Building age": ("age_band", AGE_LABELS),
    "Distance from storm track": ("distance_band", DIST_LABELS),
}
