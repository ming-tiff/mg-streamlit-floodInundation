import streamlit as st
import geopandas as gpd
import rasterio
import numpy as np
import folium
from folium.plugins import Fullscreen
from rasterio.features import shapes
from shapely.geometry import shape
from io import BytesIO
import zipfile
import tempfile
import os
import base64

st.set_page_config(page_title="Flood Inundation Mapping", layout="wide")

st.title("🌊 Flood Inundation Mapping App")

# -----------------------------
# Helper Functions
# -----------------------------

def read_dem(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    with rasterio.open(tmp_path) as src:
        dem = src.read(1, masked=True)
        transform = src.transform
        crs = src.crs
    return dem, transform, crs

def read_river_vector(uploaded_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = os.path.join(tmpdir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # ✅ If it's a ZIP shapefile, extract it
        if uploaded_file.name.endswith(".zip"):
            with zipfile.ZipFile(temp_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)
            for file in os.listdir(tmpdir):
                if file.endswith(".shp"):
                    shp_path = os.path.join(tmpdir, file)
                    gdf = gpd.read_file(shp_path)
                    break
        else:
            gdf = gpd.read_file(temp_path)

        # ✅ If CRS missing, set to Malaysia GDM2000 / BRSO
        if gdf.crs is None:
            st.warning("River shapefile CRS missing — assuming EPSG:3375 (BRSO / GDM2000).")
            gdf = gdf.set_crs("EPSG:3375")

        return gdf

def compute_flood_inundation(dem, threshold):
    """Compute flood mask for given water level (in meters)."""
    inundation = np.where(dem <= threshold, 1, 0).astype(np.uint8)
    return inundation

def raster_to_vector(mask, transform, crs):
    """Convert inundation raster to vector polygons."""
    polygons = []
    for shp, val in shapes(mask, mask=mask.astype(bool), transform=transform):
        if val == 1:
            polygons.append(shape(shp))
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    return gdf

def gdf_to_zip_download(gdf):
    """Convert GeoDataFrame to zipped Shapefile for download."""
    with tempfile.TemporaryDirectory() as tmpdir:
        shp_path = os.path.join(tmpdir, "flood_area.shp")
        gdf.to_file(shp_path)
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            for root, _, files in os.walk(tmpdir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.basename(file_path))
        zip_buffer.seek(0)
        b64 = base64.b64encode(zip_buffer.read()).decode()
        href = f'<a href="data:application/zip;base64,{b64}" download="flood_area.zip">📥 Download Flood Shapefile</a>'
        return href

# -----------------------------
# Inputs
# -----------------------------
uploaded_dem = st.file_uploader("📂 Upload DEM File (GeoTIFF)", type=["tif", "tiff"])
uploaded_river = st.file_uploader("📂 Upload River Area File (.zip or .shp)", type=["zip", "shp"])

flood_levels = [0.1, 0.3, 0.5, 1.0, 1.5]

if uploaded_dem and uploaded_river:
    dem, transform, dem_crs = read_dem(uploaded_dem)
    river_gdf = read_river_vector(uploaded_river)

    # ✅ Reproject DEM CRS to match vector CRS if different
    if dem_crs and river_gdf.crs and dem_crs != river_gdf.crs:
        st.info("Reprojecting river layer to match DEM CRS...")
        river_gdf = river_gdf.to_crs(dem_crs)

    st.success("✅ Files successfully loaded!")

    # -----------------------------
    # Flood Computation
    # -----------------------------
    st.subheader("🌧 Flood Simulation Levels")

    for lvl in flood_levels:
        st.markdown(f"**Flood Level: {lvl} m**")
        mask = compute_flood_inundation(dem, threshold=np.nanmin(dem) + lvl)
        flood_gdf = raster_to_vector(mask, transform, dem_crs)
        flood_gdf["Flood_m"] = lvl

        # ✅ Add basemap + visualization
        m = folium.Map(location=[5.98, 116.07], zoom_start=12, tiles="OpenStreetMap")
        Fullscreen().add_to(m)

        # Add flood polygons
        folium.GeoJson(
            flood_gdf.to_crs(epsg=4326),
            style_function=lambda x: {"color": "blue", "fillOpacity": 0.4},
            tooltip=folium.GeoJsonTooltip(fields=["Flood_m"])
        ).add_to(m)

        # Add river
        folium.GeoJson(
            river_gdf.to_crs(epsg=4326),
            style_function=lambda x: {"color": "cyan", "weight": 2},
            tooltip="River"
        ).add_to(m)

        folium.LayerControl().add_to(m)
        st.components.v1.html(m._repr_html_(), height=500)

        # ✅ Shapefile download button
        st.markdown(gdf_to_zip_download(flood_gdf), unsafe_allow_html=True)
