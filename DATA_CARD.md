# Data Card

## Region
China Northwest (Hexi Corridor - Northern Ningxia - Western Shaanbei)

## Data Sources
1. NOAA NCEI Global Hourly (priority)
2. Open-Meteo Archive (background and fallback)
3. Open-Meteo Elevation

## Data Types
- Dynamic station hourly sequence
- Background field hourly sequence
- Static geographic/source proxy features

## Labeling
- Wind regression target (1h/3h/6h)
- Risk class (0/1/2/3)
- Warning crossing label
- Event proxies exported at report stage

## Versioning
- Raw manifest: `data/raw/download_manifest.json`
- Processed meta: `data/processed/dataset_meta.json`

## Known Gaps
- Some stations use fallback dynamics when NOAA is unavailable.
- Remote-sensing AOD/NDVI is not fully merged yet; extension hooks reserved.
