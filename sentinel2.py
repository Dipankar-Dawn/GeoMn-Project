import os
import ee
import geemap

EE_PROJECT = "moilai"
EE_KEY_FILE = "/etc/secrets/earthengine-service-account.json"

if os.path.exists(EE_KEY_FILE):
    # Render / production
    credentials = ee.ServiceAccountCredentials(
        None,
        key_file=EE_KEY_FILE
    )
    ee.Initialize(
        credentials=credentials,
        project=EE_PROJECT
    )
    print("Earth Engine initialized using service account.")
else:
    # Local development
    ee.Initialize(project=EE_PROJECT)
    print("Earth Engine initialized using local credentials.")

#print("Earth Engine ready!")

roi = ee.Geometry.Rectangle([
    80.215, 21.825,
    80.250, 21.870
])

Map = geemap.Map()

Map.centerObject(roi, 11)

Map.addLayer(
    roi,
    {"color": "red"},
    "Test ROI"
)

Map

sentinel = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(roi)
    .filterDate("2025-01-01", "2026-06-30")
    .filter(
        ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 15)
    )
)

image = sentinel.median().clip(roi)

Map.addLayer(
    image,
    {
        "bands": ["B4", "B3", "B2"],
        "min": 0,
        "max": 3000
    },
    "Sentinel-2 RGB"
)

Map

# Ferrous Iron Ratio
ferrous_ratio = (
    image.select("B11")
    .divide(image.select("B8"))
    .rename("ferrous_ratio")
)

# Clay Mineral Index
clay_index = (
    image.select("B11")
    .divide(image.select("B12"))
    .rename("clay_index")
)

# Iron Oxide Ratio
iron_oxide_ratio = (
    image.select("B4")
    .divide(image.select("B2"))
    .rename("iron_oxide_ratio")
)

# Custom Manganese Spectral Proxy
mn_proxy = (
    image.select("B11")
    .add(image.select("B4"))
    .divide(
        image.select("B8")
        .add(image.select("B2"))
    )
    .rename("mn_proxy")
)

stats = mn_proxy.reduceRegion(
    reducer=ee.Reducer.percentile([5, 25, 50, 75, 95]),
    geometry=roi,
    scale=20,
    maxPixels=1e9
)

#print(stats.getInfo())  95th percentile gives aroundn 1.472

mn_anomaly = mn_proxy.gt(1.472562) #intentional hard-coding to be modified
Map.addLayer(
    mn_anomaly.selfMask(),
    {
        "palette": ["red"]
    },
    "Mn Proxy > 95th Percentile"
)

Map

ndvi = (
    image.select("B8")
    .subtract(image.select("B4"))
    .divide(
        image.select("B8")
        .add(image.select("B4"))
    )
    .rename("NDVI")
)

Map.addLayer(
    ndvi,
    {
        "min": -0.2,
        "max": 0.8,
        "palette": [
            "brown",
            "yellow",
            "green"
        ]
    },
    "NDVI"
)

Map

non_vegetated = ndvi.lt(0.4)

mn_anomaly_filtered = mn_anomaly.And(non_vegetated)

Map.addLayer(
    mn_anomaly_filtered.selfMask(),
    {
        "palette": ["red"]
    },
    "Mn Anomaly + NDVI Filter"
)

Map

Map.addLayer(
    image,
    {
        "bands": ["B11", "B8", "B4"],
        "min": 0,
        "max": 3500
    },
    "SWIR-NIR-Red False Color"
)

Map

Map.to_html("moil_manganese_map.html")



