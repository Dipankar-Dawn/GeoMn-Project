import ee
import geemap


# =========================================================
# EARTH ENGINE INITIALIZATION
# =========================================================

def initialize_earth_engine():

    try:

        ee.Initialize(project="moilai")

        print("✓ Earth Engine ready!")

        return True

    except Exception as e:

        print("✗ Earth Engine initialization failed:", e)

        return False


# =========================================================
# GENERATE MANGANESE PROSPECTIVITY MAP
# =========================================================

def generate_manganese_map(

    west,
    south,
    east,
    north,

    output_file="moil_manganese_map.html"

):

    try:

        # =================================================
        # REGION OF INTEREST
        # =================================================

        roi = ee.Geometry.Rectangle([

            west,
            south,
            east,
            north

        ])


        # =================================================
        # CREATE MAP
        # =================================================

        Map = geemap.Map()

        Map.centerObject(
            roi,
            11
        )


        Map.addLayer(

            roi,

            {
                "color": "red"
            },

            "Analysis Region"

        )


        # =================================================
        # SENTINEL-2 COLLECTION
        # =================================================

        sentinel = (

            ee.ImageCollection(
                "COPERNICUS/S2_SR_HARMONIZED"
            )

            .filterBounds(
                roi
            )

            .filterDate(
                "2025-01-01",
                "2026-06-30"
            )

            .filter(

                ee.Filter.lt(
                    "CLOUDY_PIXEL_PERCENTAGE",
                    15
                )

            )

        )


        # =================================================
        # MEDIAN COMPOSITE
        # =================================================

        image = (

            sentinel
            .median()
            .clip(roi)

        )


        # =================================================
        # SENTINEL-2 RGB
        # =================================================

        Map.addLayer(

            image,

            {

                "bands": [

                    "B4",
                    "B3",
                    "B2"

                ],

                "min": 0,

                "max": 3000

            },

            "Sentinel-2 RGB"

        )


        # =================================================
        # FERROUS IRON RATIO
        # =================================================

        ferrous_ratio = (

            image
            .select("B11")

            .divide(
                image.select("B8")
            )

            .rename(
                "ferrous_ratio"
            )

        )


        # =================================================
        # CLAY MINERAL INDEX
        # =================================================

        clay_index = (

            image
            .select("B11")

            .divide(
                image.select("B12")
            )

            .rename(
                "clay_index"
            )

        )


        # =================================================
        # IRON OXIDE RATIO
        # =================================================

        iron_oxide_ratio = (

            image
            .select("B4")

            .divide(
                image.select("B2")
            )

            .rename(
                "iron_oxide_ratio"
            )

        )


        # =================================================
        # CUSTOM MANGANESE SPECTRAL PROXY
        # =================================================

        mn_proxy = (

            image
            .select("B11")

            .add(
                image.select("B4")
            )

            .divide(

                image
                .select("B8")

                .add(
                    image.select("B2")
                )

            )

            .rename(
                "Mn_Proxy"
            )

        )


        # =================================================
        # CALCULATE DYNAMIC 95th PERCENTILE
        # =================================================

        stats = (

            mn_proxy
            .reduceRegion(

                reducer=
                ee.Reducer.percentile(
                    [95]
                ),

                geometry=roi,

                scale=20,

                maxPixels=1e9

            )

        )


        threshold = ee.Number(

            stats.get(
                "Mn_Proxy_p95"
            )

        )


        # =================================================
        # MANGANESE ANOMALY
        # =================================================

        mn_anomaly = (

            mn_proxy
            .gt(
                threshold
            )

        )


        # =================================================
        # NDVI
        # =================================================

        ndvi = (

            image
            .select("B8")

            .subtract(
                image.select("B4")
            )

            .divide(

                image
                .select("B8")

                .add(
                    image.select("B4")
                )

            )

            .rename(
                "NDVI"
            )

        )


        # =================================================
        # NDVI VEGETATION FILTER
        # =================================================

        non_vegetated = (

            ndvi
            .lt(0.4)

        )


        mn_anomaly_filtered = (

            mn_anomaly
            .And(
                non_vegetated
            )

        )


        # =================================================
        # ADD MANGANESE ANOMALY LAYER
        # =================================================

        Map.addLayer(

            mn_anomaly_filtered
            .selfMask(),

            {

                "palette": [
                    "red"
                ]

            },

            "Potential Mn Spectral Anomaly"

        )


        # =================================================
        # ADD NDVI LAYER
        # =================================================

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


        # =================================================
        # FALSE COLOR LAYER
        # =================================================

        Map.addLayer(

            image,

            {

                "bands": [

                    "B11",
                    "B8",
                    "B4"

                ],

                "min": 0,

                "max": 3500

            },

            "SWIR-NIR-Red False Color"

        )


        # =================================================
        # SAVE MAP AS HTML
        # =================================================

        Map.to_html(
            output_file
        )


        print(
            "✓ Manganese map generated:",
            output_file
        )


        # =================================================
        # RETURN RESULT
        # =================================================

        return {

            "success": True,

            "map_file":
                output_file,

            "threshold":
                threshold.getInfo()

        }


    except Exception as e:

        print(
            "✗ Sentinel processing error:",
            e
        )


        return {

            "success": False,

            "error":
                str(e)

        }


# =========================================================
# TEST MODE
# =========================================================

if __name__ == "__main__":

    initialized = initialize_earth_engine()


    if initialized:

        result = generate_manganese_map(

            west=80.215,

            south=21.825,

            east=80.250,

            north=21.870,

            output_file="moil_manganese_map.html"

        )


        print(result)
