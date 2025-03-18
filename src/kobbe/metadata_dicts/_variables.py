

var_dict = {
    "TIME": {
        'coverage_content_type': "coordinate",
    },
    "SAMPLE": {
        'coverage_content_type': "coordinate",
    },
    "VEL_BIN": {
        'coverage_content_type': "coordinate",
    },
    "BIN_DEPTH": {
        'coverage_content_type': "physicalMeasurement",
    },

    "SEA_ICE_DRAFT": {
        "standard_name": "sea_ice_draft",
        "units": "m",
        "long_name": "Sea ice draft estimate",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -0.5,
        "valid_max": 20.0,
    },
    "SEA_ICE_DRAFT_PING": {
        "standard_name": "sea_ice_draft",
        "units": "m",
        "long_name": "Sea ice draft estimate",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -0.5,
        "valid_max": 20.0,
    },
    "SEA_ICE_DRAFT_MEDIAN": {
        "standard_name": "sea_ice_draft",
        "units": "m",
        "long_name": "Sea ice draft estimate (median for each sample cycle)",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -0.5,
        "valid_max": 20.0,
    },
    "SEA_ICE_DRAFT_LE": {
        "standard_name": "sea_ice_draft",
        "units": "m",
        "long_name": "Sea ice draft estimate based on LE algorithm",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -0.5,
        "valid_max": 20.0,
    },
    "SEA_ICE_DRAFT_MEDIAN_LE": {
        "standard_name": "sea_ice_draft",
        "units": "m",
        "long_name": ("Sea ice draft estimate based on LE algorithm",
                      "(median for each sample cycle)"),
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -0.5,
        "valid_max": 20.0,
    },
    "SEA_ICE_DRAFT_AST": {
        "standard_name": "sea_ice_draft",
        "units": "m",
        "long_name": "Sea ice draft estimate based on AST algorithm",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -0.5,
        "valid_max": 20.0,
    },
    "SEA_ICE_DRAFT_MEDIAN_AST": {
        "standard_name": "sea_ice_draft",
        "units": "m",
        "long_name": ("Sea ice draft estimate based on AST algorithm"
                      " (median for each sample cycle)"),
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -0.5,
        "valid_max": 20.0,
    },
    "SEA_ICE_FRACTION": {
        "units": "1",
        "long_name": "Fraction of samples classified as sea ice",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 0.0,
        "valid_max": 1.0,
    },
    'ICE_IN_SAMPLE': {
        'units': '1',
        "coverage_content_type": "auxiliaryInformation",
    },
    "UICE": {
        "standard_name": "eastward_sea_ice_velocity",
        "units": "m s-1",
        "long_name": "Eastward sea ice drift velocity",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -1.5,
        "valid_max": 1.5,
    },
    "VICE": {
        "standard_name": "northward_sea_ice_velocity",
        "units": "m s-1",
        "long_name": "Northward sea ice drift velocity",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -1.5,
        "valid_max": 1.5,
    },
    "UCUR": {
        "standard_name": "eastward_sea_water_velocity",
        "units": "m s-1",
        "long_name": "Eastward sea water velocity",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -1.5,
        "valid_max": 1.5,
    },
    "VCUR": {
        "standard_name": "northward_sea_water_velocity",
        "units": "m s-1",
        "long_name": "Northward sea water velocity",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -1.5,
        "valid_max": 1.5,
    },
}

variable_order = [
    'TIME', 'SAMPLE', 'VEL_BIN',
    'SEA_ICE_DRAFT_MEDIAN',
    'SEA_ICE_DRAFT_PING',
    "SEA_ICE_DRAFT_MEDIAN_LE", "SEA_ICE_DRAFT_LE",
    "SEA_ICE_DRAFT_MEDIAN_AST", "SEA_ICE_DRAFT_AST",
    "SEA_ICE_FRACTION",
    "UICE", "VICE", "uice", "vice",
    'UCUR', 'VCUR', "ucur", "vcur",
    'ICE_IN_SAMPLE',
    'SCATTERING_SURFACE_DEPTH',
    "LATITUDE", "LONGITUDE",
    'BIN_DEPTH',
    'INSTRUMENT',
]