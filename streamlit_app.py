import streamlit as st
import rasterio
import geopandas as gpd
import numpy as np
import folium
import tempfile
import os
import zipfile
from shapely.geometry import shape
from rasterio.features import shapes
from rasterio import mask
from scipy.ndimage import binary_dilation, generate_binary_structure
from streamlit_folium import st_folium
from io import BytesIO

st.set_page_config(page_title="Flood Inundation Mapping", layout="wide")

st.title("🌊 Flood Inundation Mapping Tool")
st.markdown("""
Upload a DEM and a river shapefile (or GeoJSON).  
Then simulate flooding at selected water levels and visualize it on an interactive basemap.
""")

# --- Helper functions ---

def read_river_vector(uploaded_file):
    """Reads shapefile (even zipped) or GeoJSON into GeoDataFrame."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.read())

    if uploaded_file.name.endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        shp_files = [os.path.join(root, f) for root, _, files in os.walk(tmpdir)
                     for f in files if f.endswith(".shp")]
        if not shp_files:
            st.error("No .shp file found inside ZIP.")
            return None
        gdf = gpd.read_file(shp_files[0])
    else:
        gdf = gpd.read_file(path)

    if gdf.crs is None:
        st.warning("River file has no CRS; assuming EPSG:4326 (WGS84).")
        gdf.set_crs(epsg=4326, inplace=True)

    return gdf


def compute_connected_inundation(dem_arr, river_mask, transform, water_level):
    flooded = (dem_arr <= water_level)
    struct = generate_binary_structure(2, 2)
    connected = binary_dilation(river_mask, structure=struct, iterations=500)
    connected_flood = flooded & connected
    return connected_flood


def raster_to_vector(flood_mask, transform, level):
    results = []
    for geom, val in shapes(flood_mask.astype(np.uint8), transform=transform):
        if val == 1:
            results.append(shape(geom))
    if not results:
        return gpd.GeoDataFrame(columns=["geometry", "flood_level", "area_m2", "depth_m"], geometry="geometry", crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(geometry=results)
    gdf["flood_level"] = level
    gdf["area_m2"] = gdf.geometry.area
    gdf["depth_m"] = level
    gdf.set_crs(epsg=4326, inplace=True)
    return gdf


def add_basemap(map_object):
    folium.TileLayer('OpenStreetMap', name='Base Map').add_to(map_object)
    folium.LayerControl().add_to(map_object)
    return map_object


# --- Upload inputs ---
dem_file = st.file_uploader("Upload DEM (GeoTIFF or .asc or .nc)", type=["tif", "tiff", "asc", "nc"])
river_file = st.file_uploader("Upload River shapefile (.zip) or GeoJSON", type=["zip", "geojson", "json"])

levels_str = st.text_input("Flood levels (m, comma-separated):", "0.1,0.3,0.5,1.0,1.5")
levels = [float(x.strip()) for x in levels_str.split(",") if x.strip()]

if dem_file and river_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_dem:
        tmp_dem.write(dem_file.read())
        tmp_dem_path = tmp_dem.name

    river_gdf = read_river_vector(river_file)

    with rasterio.open(tmp_dem_path) as src:
        dem_arr = src.read(1)
        transform = src.transform
        dem_crs = src.crs

    if river_gdf.crs != dem_crs:
        try:
            river_gdf = river_gdf.to_crs(dem_crs)
        except Exception as e:
            st.error(f"Failed to align CRS: {e}")
            st.stop()

    # Rasterize river
    from rasterio.features import rasterize
    river_mask = rasterize(
        [(geom, 1) for geom in river_gdf.geometry],
        out_shape=dem_arr.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    all_floods = []

    st.subheader("🧮 Running inundation simulation...")
    progress = st.progress(0)
    for i, lvl in enumerate(levels):
        flood_mask = compute_connected_inundation(dem_arr, river_mask, transform, lvl)
        gdf = raster_to_vector(flood_mask, transform, lvl)
        if dem_crs and dem_crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        all_floods.append(gdf)
        progress.progress((i + 1) / len(levels))

    if all_floods:
        merged = gpd.GeoDataFrame(pd.concat(all_floods, ignore_index=True), crs="EPSG:4326")

        # Visualization
        st.subheader("🗺️ Flood Inundation Map")
        m = folium.Map(location=[river_gdf.to_crs(4326).geometry.centroid.y.mean(),
                                 river_gdf.to_crs(4326).geometry.centroid.x.mean()],
                       zoom_start=11)
        add_basemap(m)
        folium.GeoJson(river_gdf.to_crs(4326), name="River", style_function=lambda x: {'color': 'blue'}).add_to(m)

        colors = ["#cceeff", "#99ccff", "#66b2ff", "#3399ff", "#0077cc"]
        for i, lvl in enumerate(levels):
            subset = merged[merged["flood_level"] == lvl]
            color = colors[i % len(colors)]
            folium.GeoJson(
                subset,
                name=f"{lvl} m flood",
                style_function=lambda x, color=color: {'fillColor': color, 'color': color, 'fillOpacity': 0.5},
                tooltip=folium.GeoJsonTooltip(fields=["flood_level", "area_m2", "depth_m"])
            ).add_to(m)

        folium.LayerControl().add_to(m)
        st_folium(m, width=900, height=600)

        # Export section
        st.subheader("📦 Export flood inundation results")
        tmpdir = tempfile.mkdtemp()
        out_gpkg = os.path.join(tmpdir, "flood_inundation.gpkg")
        merged.to_file(out_gpkg, driver="GPKG")

        out_shp_dir = os.path.join(tmpdir, "flood_inundation_shp")
        os.makedirs(out_shp_dir, exist_ok=True)
        merged.to_file(out_shp_dir, driver="ESRI Shapefile")

        zip_path = os.path.join(tmpdir, "flood_inundation_shp.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(out_shp_dir):
                for f in files:
                    zipf.write(os.path.join(root, f), arcname=f)

        with open(zip_path, "rb") as f:
            st.download_button(
                "⬇️ Download Shapefile (ZIP)",
                data=f,
                file_name="flood_inundation_shp.zip",
                mime="application/zip"
            )

        with open(out_gpkg, "rb") as f:
            st.download_button(
                "⬇️ Download GeoPackage (.gpkg)",
                data=f,
                file_name="flood_inundation.gpkg",
                mime="application/geopackage+sqlite3"
            )
