import streamlit as st
import geopandas as gpd
import folium
from folium.plugins import MeasureControl
from shapely.geometry import Polygon
import tempfile
import zipfile
import os
from streamlit_folium import st_folium

st.set_page_config(page_title="Flood Inundation Mapping", layout="wide")

st.title("🌊 Flood Inundation Mapping App")
st.markdown("Upload DEM and river shapefile to simulate and visualize flood inundation areas.")

# --- Upload Section ---
st.sidebar.header("🗂️ Upload Data")
dem_file = st.sidebar.file_uploader("Upload DEM file (GeoTIFF)", type=["tif", "tiff"])
river_file = st.sidebar.file_uploader("Upload River shapefile (.zip)", type=["zip"])

# --- Flood level input ---
flood_level = st.sidebar.slider("Set Flood Water Level (m)", 0.0, 20.0, 5.0, 0.5)

# --- Temporary working directory ---
tmpdir = tempfile.mkdtemp()

# --- Function to read shapefile (supports zipped upload) ---
def read_shapefile(uploaded_file):
    temp_path = os.path.join(tmpdir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file.name.endswith(".zip"):
        with zipfile.ZipFile(temp_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        shp_path = None
        for file in os.listdir(tmpdir):
            if file.endswith(".shp"):
                shp_path = os.path.join(tmpdir, file)
                break
        if shp_path is None:
            st.error("No .shp file found inside the ZIP.")
            return None
        return gpd.read_file(shp_path)
    else:
        return gpd.read_file(temp_path)

# --- Main Processing ---
if river_file:
    river_gdf = read_shapefile(river_file)
    if river_gdf is not None:
        # Handle CRS
        if river_gdf.crs is None:
            st.warning("CRS not detected. Assigning RSO Malaya / GDM2000 (EPSG:3375).")
            river_gdf.set_crs(epsg=3375, inplace=True)

        # Reproject to WGS84 for map visualization
        try:
            river_wgs84 = river_gdf.to_crs(epsg=4326)
        except Exception as e:
            st.error(f"CRS transformation failed: {e}")
            st.stop()

        # Simulate flood polygon (for demonstration)
        bounds = river_gdf.total_bounds
        minx, miny, maxx, maxy = bounds
        buffer_distance = flood_level * 100  # 100m per meter of flood, for visualization
        flood_poly = Polygon([
            (minx - buffer_distance, miny - buffer_distance),
            (minx - buffer_distance, maxy + buffer_distance),
            (maxx + buffer_distance, maxy + buffer_distance),
            (maxx + buffer_distance, miny - buffer_distance)
        ])
        flood_gdf = gpd.GeoDataFrame([{"geometry": flood_poly, "flood_m": flood_level}],
                                     crs=river_gdf.crs)
        flood_wgs84 = flood_gdf.to_crs(epsg=4326)

        # --- Folium Map ---
        st.subheader("🗺️ Flood Inundation Map")

        # Create folium map centered on data
        m = folium.Map(
            location=[river_wgs84.geometry.centroid.y.mean(), river_wgs84.geometry.centroid.x.mean()],
            zoom_start=10,
            tiles="CartoDB positron"
        )

        folium.TileLayer('OpenStreetMap').add_to(m)
        folium.TileLayer('Stamen Terrain').add_to(m)

        # Add layers
        folium.GeoJson(
            river_wgs84,
            name='River',
            tooltip=folium.GeoJsonTooltip(fields=[],
                                          aliases=[],
                                          labels=False)
        ).add_to(m)

        folium.GeoJson(
            flood_wgs84,
            name='Flood Inundation',
            style_function=lambda x: {
                'fillColor': 'blue',
                'color': 'blue',
                'weight': 1,
                'fillOpacity': 0.4
            },
            tooltip=folium.GeoJsonTooltip(fields=['flood_m'], aliases=['Flood depth (m):'])
        ).add_to(m)

        folium.LayerControl().add_to(m)
        m.add_child(MeasureControl())

        # Display map
        st_map = st_folium(m, width=1100, height=600)

        # --- Download Shapefile ---
        output_shp = os.path.join(tmpdir, "flood_inundation.shp")
        flood_gdf.to_file(output_shp)

        # Zip the shapefile
        zip_path = os.path.join(tmpdir, "flood_inundation.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for file in os.listdir(tmpdir):
                if file.startswith("flood_inundation"):
                    zipf.write(os.path.join(tmpdir, file), file)

        with open(zip_path, "rb") as f:
            st.download_button("⬇️ Download Flood Shapefile", f, "flood_inundation.zip")

else:
    st.info("Please upload at least the river shapefile (.zip) to begin.")
