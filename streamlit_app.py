"""
Streamlit app: Flood inundation mapping from DEM + river vector
- Supports DEM in GeoTIFF / ASCII grid (.asc) / NetCDF (with z variable) / other GDAL-readable rasters
- Accepts river vector as Shapefile (.shp + .shx + .dbf) or GeoJSON
- Computes inundation polygons for a set of water levels (default: 0.1, 0.3, 0.5, 1.0, 1.5 m)
- Keeps only flood extents hydraulically connected to the river (connectivity by pixel)
- Produces styled map preview and downloads: GeoPackage (.gpkg) and zipped ESRI Shapefile

Dependencies:
pip install rasterio geopandas fiona numpy scipy shapely rasterio[xarray] xarray rioxarray streamlit streamlit_folium folium matplotlib pyproj

Run:
streamlit run flood_inundation_streamlit_app.py

Notes:
- Sentinel-1 imagery is SAR backscatter and is NOT a DEM. To use Sentinel-1 you must derive or provide a DEM first (e.g., from SRTM, TanDEM-X, or copernicus DEM).
- This is a simple static water-level inundation (bathtub) model with connectivity to river; it does NOT perform hydraulic routing (no 1D/2D flow simulation). For hydraulic modeling use HEC-RAS, LISFLOOD-FP, or 2D hydrodynamic models.
"""

import streamlit as st
from streamlit_folium import st_folium
import rasterio
from rasterio.features import shapes
from rasterio import Affine
import geopandas as gpd
import numpy as np
import tempfile
import os
import zipfile
from shapely.geometry import shape, mapping
from scipy import ndimage
import json
import folium
from folium.plugins import BeautifyIcon
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Flood Inundation Mapper")

DEFAULT_LEVELS = [0.1, 0.3, 0.5, 1.0, 1.5]

st.title("Flood Inundation Mapper — DEM + River → Flood Polygons")

with st.sidebar:
    st.header("Inputs")
    dem_file = st.file_uploader("Upload DEM (GeoTIFF, .asc, NetCDF...)", type=["tif","tiff","asc","nc","grd"], help='ASCII .asc should be ESRI ASCII grid; NetCDF should contain a single 2D elevation variable')
    river_file = st.file_uploader("Upload river vector (shp as zip or geojson)", type=["zip","geojson"], help='For shapefile, zip all shapefile components (.shp,.shx,.dbf) into a .zip before upload')
    levels_input = st.text_input("Water levels (m) — comma separated", value=",".join(map(str, DEFAULT_LEVELS)))
    run_button = st.button("Run inundation analysis")


def save_uploaded_file(uploaded, dest_path):
    with open(dest_path, 'wb') as f:
        f.write(uploaded.getbuffer())


def read_raster_from_fileobj(path):
    # rasterio can open files from path
    src = rasterio.open(path)
    arr = src.read(1, masked=True).astype('float32')
    transform = src.transform
    crs = src.crs
    nodata = src.nodata
    return arr, transform, crs, nodata


def read_river_vector(path_or_dir):
    # If it's a zip containing shapefile, geopandas can read from zip
    try:
        gdf = gpd.read_file(path_or_dir)
        return gdf
    except Exception as e:
        raise e


def rasterize_geometry_mask(geoms, out_shape, transform):
    # return boolean mask where river pixels = True
    import rasterio.features
    shapes_gen = ((mapping(g), 1) for g in geoms)
    mask = rasterio.features.rasterize(shapes_gen, out_shape=out_shape, transform=transform, fill=0, dtype='uint8')
    return mask.astype(bool)


def compute_connected_inundation(dem, river_mask, transform, level):
    """
    dem: 2D numpy array (masked array recommended)
    river_mask: boolean mask of river pixels
    level: water level (float)
    Returns boolean mask of inundation connected to river
    """
    # Create flooded candidate mask where DEM <= level
    flooded_candidate = (dem <= level) & (~np.isnan(dem))

    if not flooded_candidate.any():
        return np.zeros(dem.shape, dtype=bool)

    # Label connected components in flooded_candidate
    structure = np.ones((3,3), dtype=np.int)  # 8-connectivity
    labeled, ncomp = ndimage.label(flooded_candidate, structure=structure)

    # Find labels that intersect river_mask
    river_labels = np.unique(labeled[river_mask & (labeled>0)])
    if len(river_labels) == 0:
        # No direct contact: optionally expand river_mask by small distance? For now, return empty
        return np.zeros(dem.shape, dtype=bool)

    # create final mask
    connected_mask = np.isin(labeled, river_labels)
    return connected_mask


def mask_to_polygons(mask, transform, crs, min_area_m2=1.0):
    # mask: boolean 2D array
    results = []
    for geom_dict, val in shapes(mask.astype('uint8'), mask=mask, transform=transform):
        if val == 1:
            geom = shape(geom_dict)
            if geom.is_valid and geom.area >= 0:
                results.append(geom)
    if not results:
        return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=crs)
    gdf = gpd.GeoDataFrame(geometry=results, crs=crs)
    # dissolve to single polygons and remove tiny areas
    gdf = gdf.dissolve(by=lambda x: 0)
    gdf = gdf.explode(ignore_index=True)
    if 'geometry' in gdf:
        gdf = gdf[gdf.geometry.area >= min_area_m2]
    return gdf


def create_colormap(levels):
    # return dictionary level -> rgba color (light->dark blues)
    cmap = plt.get_cmap('Blues')
    n = len(levels)
    colors = [plt.cm.Blues(0.4 + 0.5*i/(n-1)) for i in range(n)] if n>1 else [plt.cm.Blues(0.6)]
    return {lvl: colors[i] for i,lvl in enumerate(sorted(levels))}


def make_map(center, dem_path=None, river_gdf=None, flood_gdfs=None, colormap=None):
    m = folium.Map(location=center, zoom_start=12, tiles='CartoDB.Positron')
    if river_gdf is not None and not river_gdf.empty:
        folium.GeoJson(river_gdf.to_crs(epsg=4326), name='River', tooltip='River').add_to(m)
    if flood_gdfs:
        for lvl, gdf in flood_gdfs.items():
            if gdf is None or gdf.empty:
                continue
            color = colormap.get(lvl, (0,0,1,0.5))
            hexc = mpl_color_to_hex(color)
            folium.GeoJson(gdf.to_crs(epsg=4326), name=f'Flood {lvl} m', style_function=lambda feat, hexc=hexc: {'"' + 'fillColor' + '"': hexc, '"' + 'color' + '"': hexc, '"' + 'weight' + '"': 0.5, '"' + 'fillOpacity' + '"':0.5}).add_to(m)
    folium.LayerControl().add_to(m)
    return m


def mpl_color_to_hex(c):
    # c is RGBA tuple with floats
    try:
        import matplotlib
        return matplotlib.colors.to_hex(c)
    except Exception:
        return '#0000ff'


if run_button:
    if dem_file is None or river_file is None:
        st.error("Please upload both a DEM and a river vector file.")
    else:
        with st.spinner('Processing...'):
            tmpdir = tempfile.mkdtemp()
            dem_path = os.path.join(tmpdir, dem_file.name)
            save_uploaded_file(dem_file, dem_path)

            # Save river file
            river_path = os.path.join(tmpdir, river_file.name)
            save_uploaded_file(river_file, river_path)

            # If river zip, geopandas can read 'zip://path'
            if river_file.name.lower().endswith('.zip'):
                river_read_path = f'zip://{river_path}'
            else:
                river_read_path = river_path

            try:
                dem_arr, transform, crs, nodata = read_raster_from_fileobj(dem_path)
            except Exception as e:
                st.exception(e)
                st.stop()

            try:
                river_gdf = read_river_vector(river_read_path)
            except Exception as e:
                st.exception(e)
                st.stop()

            # parse levels
            try:
                levels = [float(x.strip()) for x in levels_input.split(',') if x.strip()!='']
            except Exception:
                st.error('Invalid water levels input. Use comma separated numbers (e.g. 0.1,0.3,0.5)')
                st.stop()

            # rasterize river to grid
            out_shape = dem_arr.shape
            river_mask = rasterize_geometry_mask(river_gdf.geometry, out_shape, transform)

            flood_gdfs = {}
            colormap = create_colormap(levels)

            for lvl in levels:
                conn_mask = compute_connected_inundation(dem_arr, river_mask, transform, lvl)
                gdf = mask_to_polygons(conn_mask, transform, crs)
                gdf['level_m'] = lvl
                flood_gdfs[lvl] = gdf

            # combine all into single geodataframe with level attribute
            all_polys = gpd.GeoDataFrame(pd.concat([g for g in [gdf.assign(level_m=l) for l,gdf in flood_gdfs.items() if gdf is not None and not gdf.empty]], ignore_index=True), crs=crs) if any([not gdf.empty for gdf in flood_gdfs.values()]) else gpd.GeoDataFrame(columns=['geometry','level_m'], geometry='geometry', crs=crs)

            # save outputs
            out_gpkg = os.path.join(tmpdir, 'flood_inundation.gpkg')
            if not all_polys.empty:
                all_polys.to_file(out_gpkg, driver='GPKG', layer='flood_inundation')

                # Save shapefile components and zip them
                shp_dir = os.path.join(tmpdir, 'shp_out')
                os.makedirs(shp_dir, exist_ok=True)
                shp_base = os.path.join(shp_dir, 'flood_inundation.shp')
                all_polys.to_file(shp_base, driver='ESRI Shapefile')

                # zip shapefile
                zip_path = os.path.join(tmpdir, 'flood_shapefile.zip')
                with zipfile.ZipFile(zip_path, 'w') as zf:
                    for fname in os.listdir(shp_dir):
                        zf.write(os.path.join(shp_dir, fname), arcname=fname)

            # present results
            st.success('Processing finished.')
            # Map center
            try:
                bounds = river_gdf.to_crs(epsg=4326).total_bounds
                center = [(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2]
            except Exception:
                center = [0,0]

            # show map preview
            m = folium.Map(location=center, zoom_start=12, tiles='CartoDB.Positron')
            # add river
            if not river_gdf.empty:
                folium.GeoJson(river_gdf.to_crs(epsg=4326), name='River', tooltip='River').add_to(m)
            # add floods
            for lvl, gdf in flood_gdfs.items():
                if gdf is None or gdf.empty:
                    continue
                hexc = mpl_color_to_hex(colormap.get(lvl))
                folium.GeoJson(gdf.to_crs(epsg=4326), name=f'Flood {lvl} m', style_function=lambda feat, hexc=hexc: {'"' + 'fillColor' + '"': hexc, '"' + 'color' + '"': hexc, '"' + 'weight' + '"': 0.5, '"' + 'fillOpacity' + '"':0.5}).add_to(m)

            folium.LayerControl().add_to(m)
            st_folium(m, width=900)

            # provide downloads
            if not all_polys.empty:
                with open(out_gpkg, 'rb') as f:
                    st.download_button('Download GeoPackage (.gpkg)', f.read(), file_name='flood_inundation.gpkg')
                with open(zip_path, 'rb') as f:
                    st.download_button('Download zipped Shapefile', f.read(), file_name='flood_inundation_shp.zip')
            else:
                st.info('No inundation polygons produced for the given levels (no DEM cells below levels connected to river).')


st.markdown("""
### Notes & Tips
- This app models static water levels (bathtub method) with connectivity to the river (pixel-connected) but does not model hydraulics or velocity.
- For more realistic flood modelling consider HEC-RAS, LISFLOOD-FP, or full 2D models and incorporate discharge/hydrographs.
- If your DEM has vertical datum offset, ensure the water levels are referenced to the same datum.
""")
