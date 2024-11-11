var_dict = {
    "SEA_ICE_DRAFT": {
        "standard_name": "sea_ice_draft",
        "units": "m",
        "long_name": "Sea ice draft estimate",
        "coverage_content_type": "physicalMeasurement",
    },
    "SEA_ICE_DRAFT_MEDIAN": {
        "standard_name": "sea_ice_draft",
        "units": "m",
        "long_name": "Sea ice draft estimate (median for each sample cycle)",
        "coverage_content_type": "physicalMeasurement",
    },
    "SEA_ICE_DRAFT_LE": {
        "standard_name": "sea_ice_draft",
        "units": "m",
        "long_name": "Sea ice draft estimate based on LE algorithm",
        "coverage_content_type": "physicalMeasurement",
    },
    "SEA_ICE_DRAFT_MEDIAN_LE": {
        "standard_name": "sea_ice_draft",
        "units": "m",
        "long_name": ("Sea ice draft estimate based on LE algorithm",
                      "(median for each sample cycle)"),
        "coverage_content_type": "physicalMeasurement",
    },
    "SEA_ICE_DRAFT_AST": {
        "standard_name": "sea_ice_draft",
        "units": "m",
        "long_name": "Sea ice draft estimate based on AST algorithm",
        "coverage_content_type": "physicalMeasurement",
    },
    "SEA_ICE_DRAFT_MEDIAN_AST": {
        "standard_name": "sea_ice_draft",
        "units": "m",
        "long_name": ("Sea ice draft estimate based on AST algorithm"
                      " (median for each sample cycle)"),
        "coverage_content_type": "physicalMeasurement",
    },
    "SEA_ICE_FRACTION": {
        "units": "1",
        "long_name": "Fraction of samples classified as sea ice",
        "coverage_content_type": "physicalMeasurement",
    },
    "UICE": {
        "standard_name": "eastward_sea_ice_velocity",
        "units": "m s-1",
        "long_name": "Eastward sea ice draft velocity",
        "coverage_content_type": "physicalMeasurement",
    },
    "VICE": {
        "standard_name": "northward_sea_ice_velocity",
        "units": "m s-1",
        "long_name": "Northward sea ice draft velocity",
        "coverage_content_type": "physicalMeasurement",
    },
}


variable_order = [
    "SEA_ICE_DRAFT_MEDIAN", "SEA_ICE_DRAFT",
    "SEA_ICE_DRAFT_MEDIAN_LE", "SEA_ICE_DRAFT_LE",
    "SEA_ICE_DRAFT_MEDIAN_AST", "SEA_ICE_DRAFT_AST",
    "SEA_ICE_FRACTION",
    "UICE", "VICE", "uice", "vice",
    "LATITUDE", "LONGITUDE",]