import glob
import os
import geopandas as gpd
import pandas as pd
import duckdb

# Clean up existing local database builds
meta_db_path = "metadata.duckdb"
temp_db_path = "temp_processing.duckdb"

for path in [meta_db_path, temp_db_path]:
    if os.path.exists(path):
        os.remove(path)

meta_con = duckdb.connect(meta_db_path)
temp_con = duckdb.connect(temp_db_path)

meta_con.execute("""
    CREATE TABLE metadata (
        global_id VARCHAR PRIMARY KEY,
        fid BIGINT,
        code VARCHAR,
        height DOUBLE,
        h_stdev DOUBLE,
        vel DOUBLE,
        v_stdev DOUBLE,
        coherence DOUBLE,
        eff_area DOUBLE,
        latitude DOUBLE,
        longitude DOUBLE,
        inc_angle DOUBLE,
        az_angle DOUBLE,
        heading DOUBLE,
        ref_lat DOUBLE,
        ref_lon DOUBLE,
        file_name VARCHAR
    )
""")

temp_con.execute("""
    CREATE TABLE displacement (
        global_id VARCHAR,
        date DATE,
        displacement DOUBLE
    )
""")

gpkg_files = glob.glob("data/*.gpkg")
map_points_list = []
ref_points_list = []

print(f"Processing {len(gpkg_files)} GPKG files...")

for file_idx, file_path in enumerate(gpkg_files, 1):
    file_name = os.path.basename(file_path)
    print(f"[{file_idx}/{len(gpkg_files)}] Reading {file_name}...")
    
    gdf = gpd.read_file(file_path, engine="pyogrio")
    
    if 'fid' in gdf.columns:
        gdf['fid_clean'] = gdf['fid']
    elif 'FID' in gdf.columns:
        gdf['fid_clean'] = gdf['FID']
    else:
        gdf['fid_clean'] = gdf.index + 1

    gdf['global_id'] = f"F{file_idx}_" + gdf['fid_clean'].astype(str)
    gdf['file_name'] = file_name
    
    cols_lower = {c.lower(): c for c in gdf.columns}
    def get_col(name, default_val=0):
        return gdf[cols_lower[name]] if name in cols_lower else default_val

    # GeoJSON Map Points
    code_col = cols_lower.get('code')
    vel_col = cols_lower.get('vel')
    map_gdf = gpd.GeoDataFrame({
        'global_id': gdf['global_id'],
        'fid': gdf['fid_clean'],
        'CODE': gdf[code_col] if code_col else '',
        'VEL': gdf[vel_col] if vel_col else 0.0,
        'geometry': gdf['geometry']
    }, crs=gdf.crs)
    map_points_list.append(map_gdf)
    
    # Reference Locations
    ref_lat_col = cols_lower.get('ref_lat')
    ref_lon_col = cols_lower.get('ref_lon')
    ref_lat = gdf[ref_lat_col].iloc[0] if ref_lat_col else 0.0
    ref_lon = gdf[ref_lon_col].iloc[0] if ref_lon_col else 0.0
    
    ref_df = pd.DataFrame([{
        'file_name': file_name,
        'ref_lat': ref_lat,
        'ref_lon': ref_lon,
        'geometry': gpd.points_from_xy([ref_lon], [ref_lat])[0]
    }])
    ref_points_list.append(gpd.GeoDataFrame(ref_df, crs=gdf.crs))
    
    # Write Metadata
    meta_df = pd.DataFrame({
        'global_id': gdf['global_id'],
        'fid': gdf['fid_clean'],
        'code': get_col('code', ''),
        'height': get_col('height', 0.0),
        'h_stdev': get_col('h_stdev', 0.0),
        'vel': get_col('vel', 0.0),
        'v_stdev': get_col('v_stdev', 0.0),
        'coherence': get_col('coherence', 0.0),
        'eff_area': get_col('eff_area', 1.0),
        'latitude': get_col('latitude', 0.0),
        'longitude': get_col('longitude', 0.0),
        'inc_angle': get_col('inc_angle', 0.0),
        'az_angle': get_col('az_angle', 0.0),
        'heading': get_col('heading', 0.0),
        'ref_lat': ref_lat,
        'ref_lon': ref_lon,
        'file_name': file_name
    })
    meta_con.register("temp_meta", meta_df)
    meta_con.execute("INSERT INTO metadata SELECT * FROM temp_meta")
    
    # Unpivot Displacement Time Series
    date_cols = [c for c in gdf.columns if c.startswith('D20')]
    df_ts = gdf[['global_id'] + date_cols]
    df_long = df_ts.melt(id_vars=['global_id'], var_name='date', value_name='displacement')
    df_long['date'] = pd.to_datetime(df_long['date'].str.replace('D', ''), format='%Y%m%d')
    
    temp_con.register("temp_ts", df_long)
    temp_con.execute("INSERT INTO displacement SELECT global_id, date, displacement FROM temp_ts")

# Save GeoJSON Files
print("Saving GeoJSON layers...")
pd.concat(map_points_list, ignore_index=True).to_file("insar_map_points.geojson", driver="GeoJSON")
pd.concat(ref_points_list, ignore_index=True).to_file("reference_points.geojson", driver="GeoJSON")

# Export Compressed Parquet File
print("Compressing time series to 'timeseries.parquet'...")
temp_con.execute("""
    COPY (
        SELECT global_id, date, displacement 
        FROM displacement 
        ORDER BY global_id
    ) TO 'timeseries.parquet' (FORMAT PARQUET, COMPRESSION 'ZSTD');
""")

meta_con.close()
temp_con.close()
os.remove(temp_db_path)

print("Processing complete!")