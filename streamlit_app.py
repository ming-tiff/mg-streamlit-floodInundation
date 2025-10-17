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

# -------------------------------------------
# Streamlit UI
# -------------------------------------------
st.set_page_config(page_title="Flood Inundation Mapping", layout="wide")
st.title("🌊 Flood Inundation Mapping App")

# -------------------------------------------
# Helper Functions
# -------------------------------------------

def read_dem(uploaded_file):
    """Read DEM (GeoTIFF) and extract array, transform, CRS."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    with rasterio.open(tmp_path) as src:
        dem = src.read(1, masked=True)
        transform = src.transform
        crs = src.crs
    return dem, transform, crs


def read_river_vector(uploaded_file):
    """Read river shapefile (.zip or .shp). Assign CRS if missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = os.path.join(tmpdir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # If ZIP → extract shapefile
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

        # If CRS missing → set to Malaysia BRSO / GDM2000 (EPSG:3375)
        if gdf.crs is None:
            st.warning("River shapefile CRS missing — assuming EPSG:3375 (BRSO / GDM2000).")
            gdf = gdf.set_crs("EPSG:3375", allow_override=True)

        return gdf


def compute_flood_inundation(dem, threshold):
    """Create binary flood mask below threshold elevation."""
    inundation = np.where(dem <= threshold, 1, 0).astype(np.uint8)
    return inundation


def raster_to_vector(mask, transform, crs):
    """Convert binary raster flood mask to vector polygons."""
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
        with zipfile.ZipF
