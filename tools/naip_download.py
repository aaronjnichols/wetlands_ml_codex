import geopandas as gpd
from geoai.download import download_naip


# user inputs
output_dir = r"data/naip"
aoi_gpkg = r"data/train_aoi.gpkg"
max_items = 4
year = 2023

# get aoi bbox
aoi_gdf = gpd.read_file(aoi_gpkg)
aoi_wgs84 = aoi_gdf.to_crs(4326)
minx, miny, maxx, maxy = aoi_wgs84.total_bounds
aoi_bbox = (minx, miny, maxx, maxy)

# download naip
naip_paths = download_naip(
    aoi_bbox,
    output_dir=output_dir,
    year=year,
    max_items=max_items,
    overwrite=False,
    preview=False,
)

print(naip_paths)