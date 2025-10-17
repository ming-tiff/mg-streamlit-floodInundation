"""
Streamlit app: Flood inundation mapping from DEM + river vector
"""

import streamlit as st
from streamlit_folium import st_folium
import rasterio
from rasterio.features import shapes
import geopandas as gpd
import numpy as np
import tempfile
import os
import zipfile
from shapely.geometry import shape, mapping
from scipy import ndimage
import pandas as pd
import folium
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Flood Inundation Mapper")

DEFAULT_LEVELS = [0.1, 0.3, 0.5, 1.0, 1.5]

st.title("Flood Inundation Mapper — DEM + River → Flood Polygons")

with st.sidebar:
    st.header("Inputs")
    dem_file = st.file_uploader("Upload DEM (GeoTIFF, .asc, NetCDF...)", type=["tif", "tiff", "asc", "nc", "grd"])
    river_file = st.file_uploader("Upload river vector (shp as zip or geojson)", type=["zip", "geojson"])
    levels_input = st.text_input("Water levels (m) — comma separated", value=",".join(map(str, DEFAULT_LEVELS)))
    run_button = st.button("Run inundation analysis")

# =========================
# Utility functions
# =========================
def save_uploaded_file(uploaded, dest_path):
    with open(dest_path, 'wb') as f:
        f.write(uploaded.getbuffer())

def read_raster_from_fileobj(path):
    src = rasterio.open(path)
    arr = src.read(1, masked=True).astype('float32')
    transform = src.transform
    crs = src.crs
    nodata = src.nodata
    return arr, transform, crs, nodata

def extract_shapefile_from_zip(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    shp_files = [os.path.join(root, f)
                 for root, _, files in os.walk(extract_dir)
                 for f in files if f.lower().endswith('.shp')]
    if not shp_files:
        raise ValueError('No .shp file found in uploaded ZIP.')
    return shp_files[0]

def read_river_vector(path_or_dir):
    return gpd.read_file(path_or_dir)

def rasterize_geometry_mask(geoms, out_shape, transform):
    import rasterio.features
    shapes_gen = ((mapping(g), 1) for g in geoms)
    mask = rasterio.features.rasterize(shapes_gen, out_shape=out_shape, transform=transform, fill=0, dtype='uint8')
    return mask.astype(bool)

def compute_connected_inundation(dem, river_mask, transform, level):
    flooded_candidate = (dem <= level) & (~np.isnan(dem))
    if not flooded_candidate.any():
        return np.zeros(dem.shape, dtype=bool)
    structure = np.ones((3, 3), dtype=int)
    labeled, ncomp = ndimage.label(flooded_candidate, structure=structure)
    river_labels = np.unique(labeled[river_mask & (labeled > 0)])
    if len(river_labels) == 0:
        return np.zeros(dem.shape, dtype=bool)
    return np.isin(labeled, river_labels)

def mask_to_polygons(mask, transform, crs, min_area_m2=1.0):
    results = []
    for geom_dict, val in shapes(mask.astype('uint8'), mask=mask, transform=transform):
        if val == 1:
            geom = shape(geom_dict)
            if geom.is_valid and geom.area >= 0:
                results.append(geom)
    if not results:
        return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=crs)
    gdf = gpd.GeoDataFrame(geometry=results, crs=crs)
    gdf = gdf.dissolve(by=lambda x: 0).explode(ignore_index=True)
    if 'geometry' in gdf:
        gdf = gdf[gdf.geometry.area >= min_area_m2]
    return gdf

def create_colormap(levels):
    cmap = plt.get_cmap('Blues')
    n = len(levels)
    colors = [plt.cm.Blues(0.4 + 0.5*i/(n-1)) for i in range(n)] if n > 1 else [plt.cm.Blues(0.6)]
    return {lvl: colors[i] for i, lvl in enumerate(sorted(levels))}

def mpl_color_to_hex(c):
    import matplotlib
    return matplotlib.colors.to_hex(c)

# =========================
# MAIN PROCESSING SECTION
# =========================

if run_button:
    if dem_file is None or river_file is None:
        st.error("Please upload both a DEM and a river vector file.")
    else:
        with st.spinner('Processing...'):
            tmpdir = tempfile.mkdtemp()
            dem_path = os.path.join(tmpdir, dem_file.name)
            save_uploaded_file(dem_file, dem_path)

            river_path = os.path.join(tmpdir, river_file.name)
            save_uploaded_file(river_file, river_path)

            if river_file.name.lower().endswith('.zip'):
                extract_dir = os.path.join(tmpdir, 'river_extract')
                os.makedirs(extract_dir, exist_ok=True)
                shp_path = extract_shapefile_from_zip(river_path, extract_dir)
                river_read_path = shp_path
            else:
                river_read_path = river_path

# =========================
# MAIN PROCESSING SECTION
# =========================

    zip_path = None  # ✅ define globally to persist through reruns

    if run_button:
        if dem_file is None or river_file is None:
            st.error("Please upload both a DEM and a river vector file.")
        else:
            with st.spinner('Processing...'):
                ...

            dem_arr, transform, crs, nodata = read_raster_from_fileobj(dem_path)
            river_gdf = read_river_vector(river_read_path)

            levels = [float(x.strip()) for x in levels_input.split(',') if x.strip() != '']

            out_shape = dem_arr.shape
            river_mask = rasterize_geometry_mask(river_gdf.geometry, out_shape, transform)

            flood_gdfs = {}
            colormap = create_colormap(levels)

            for lvl in levels:
                conn_mask = compute_connected_inundation(dem_arr, river_mask, transform, lvl)
                gdf = mask_to_polygons(conn_mask, transform, crs)
                gdf['level_m'] = lvl
                flood_gdfs[lvl] = gdf

            valid_gdfs = [g for g in flood_gdfs.values() if not g.empty]
            if valid_gdfs:
                all_polys = gpd.GeoDataFrame(pd.concat(valid_gdfs, ignore_index=True), crs=crs)
            else:
                all_polys = gpd.GeoDataFrame(columns=['geometry', 'level_m'], geometry='geometry', crs=crs)

            out_gpkg = os.path.join(tmpdir, 'flood_inundation.gpkg')
            zip_path = None  # ✅ ensure variable exists to avoid NameError

            if not all_polys.empty:
                all_polys.to_file(out_gpkg, driver='GPKG', layer='flood_inundation')
                shp_dir = os.path.join(tmpdir, 'shp_out')
                os.makedirs(shp_dir, exist_ok=True)
                shp_base = os.path.join(shp_dir, 'flood_inundation.shp')
                all_polys.to_file(shp_base, driver='ESRI Shapefile')
                zip_path = os.path.join(tmpdir, 'flood_shapefile.zip')
                with zipfile.ZipFile(zip_path, 'w') as zf:
                    for fname in os.listdir(shp_dir):
                        zf.write(os.path.join(shp_dir, fname), arcname=fname)

            st.success('Processing finished.')

            # Create folium map
            try:
                bounds = river_gdf.to_crs(epsg=4326).total_bounds
                center = [(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2]
            except Exception:
                center = [0, 0]

            m = folium.Map(location=center, zoom_start=12, tiles='CartoDB.Positron')
            if not river_gdf.empty:
                folium.GeoJson(river_gdf.to_crs(epsg=4326), name='River', tooltip='River').add_to(m)
            for lvl, gdf in flood_gdfs.items():
                if gdf is None or gdf.empty:
                    continue
                hexc = mpl_color_to_hex(colormap.get(lvl))
                folium.GeoJson(
                    gdf.to_crs(epsg=4326),
                    name=f'Flood {lvl} m',
                    style_function=lambda feat, hexc=hexc: {
                        'fillColor': hexc, 'color': hexc, 'weight': 0.5, 'fillOpacity': 0.5
                    }
                ).add_to(m)
            folium.LayerControl().add_to(m)

            # ✅ Persist map & results
            st.session_state['flood_map'] = m
            st.session_state['gpkg_data'] = out_gpkg
            st.session_state['zip_data'] = zip_path
            st.session_state['analysis_done'] = True

# ✅ Show map & downloads persistently (even after rerun)
if st.session_state.get('analysis_done') and 'flood_map' in st.session_state:
    st_folium(st.session_state['flood_map'], width=900, key="flood_map_display")

    with open(st.session_state['gpkg_data'], 'rb') as f:
        st.download_button('Download GeoPackage (.gpkg)', f.read(), file_name='flood_inundation.gpkg')

    with open(st.session_state['zip_data'], 'rb') as f:
        st.download_button('Download zipped Shapefile', f.read(), file_name='flood_inundation_shp.zip')

st.markdown("""
### Notes & Tips
- This app models static water levels (bathtub method) with connectivity to the river (pixel-connected) but does not model hydraulics or velocity.
- For more realistic flood modelling consider HEC-RAS, LISFLOOD-FP, or full 2D models and incorporate discharge/hydrographs.
- If your DEM has vertical datum offset, ensure the water levels are referenced to the same datum.
""")
