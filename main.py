import os
import duckdb
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from huggingface_hub import hf_hub_download

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID", "rnus/jgms_sar4road_nat_200m_s1descending")

def ensure_file_exists(filename: str):
    """Downloads required dataset files from Hugging Face repo if not cached locally."""
    if not os.path.exists(filename):
        print(f"Downloading {filename} from Hugging Face repo '{HF_REPO_ID}'...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir="."
        )
        print(f"{filename} download complete!")

@app.on_event("startup")
def startup_event():
    print("Pre-downloading essential datasets...")
    ensure_file_exists("metadata.duckdb")
    ensure_file_exists("reference_points.geojson")
    ensure_file_exists("timeseries.parquet")
    print("Backend ready!")

@app.get("/")
def root():
    return {"status": "sar4road_dashboard Private Hugging Face API Online"}

@app.get("/geojson/reference_points.geojson")
def get_ref_points():
    ensure_file_exists("reference_points.geojson")
    return FileResponse("reference_points.geojson")

@app.get("/geojson/insar_map_points.geojson")
def get_insar_map_points(
    min_lat: float = Query(None),
    max_lat: float = Query(None),
    min_lon: float = Query(None),
    max_lon: float = Query(None),
    limit: int = Query(8000)
):
    ensure_file_exists("metadata.duckdb")
    con = duckdb.connect("metadata.duckdb", read_only=True)

    cols_info = con.execute("DESCRIBE metadata").fetchall()
    cols = [r[0] for r in cols_info]
    cols_lower = [c.lower() for c in cols]

    gid_col = next((c for c, cl in zip(cols, cols_lower) if cl in ['global_id', 'fid', 'code', 'id']), cols[0])
    lat_col = next((c for c, cl in zip(cols, cols_lower) if 'lat' in cl), 'latitude')
    lon_col = next((c for c, cl in zip(cols, cols_lower) if 'lon' in cl or 'lng' in cl), 'longitude')
    vel_col = next((c for c, cl in zip(cols, cols_lower) if 'vel' in cl or 'speed' in cl), 'vel')

    if None not in (min_lat, max_lat, min_lon, max_lon):
        query = f"""
            SELECT {gid_col}, {lat_col}, {lon_col}, {vel_col}
            FROM metadata 
            WHERE {lat_col} BETWEEN ? AND ? 
              AND {lon_col} BETWEEN ? AND ?
            LIMIT ?
        """
        params = [min_lat, max_lat, min_lon, max_lon, limit]
    else:
        query = f"SELECT {gid_col}, {lat_col}, {lon_col}, {vel_col} FROM metadata LIMIT 3000"
        params = [limit]

    rows = con.execute(query, params).fetchall()
    con.close()

    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(r[2]), float(r[1])]
            },
            "properties": {
                "global_id": str(r[0]),
                "velocity": float(r[3]) if r[3] is not None else 0.0
            }
        }
        for r in rows if r[1] is not None and r[2] is not None
    ]
    return {"type": "FeatureCollection", "features": features}

@app.get("/point/{global_id}")
def get_point_details(global_id: str):
    """
    Fetches point metadata and time-series by cross-matching global_id, fid, 
    and code dynamically across metadata.duckdb and timeseries.parquet.
    """
    ensure_file_exists("metadata.duckdb")
    ensure_file_exists("timeseries.parquet")

    target_id = str(global_id).strip()
    clean_id = target_id.rstrip('.0')

    metadata = {}
    dates = []
    displacement = []

    # 1. Dynamic Metadata Match (Checks global_id, fid, code, and id columns)
    try:
        m_con = duckdb.connect("metadata.duckdb", read_only=True)
        m_cols_info = m_con.execute("DESCRIBE metadata").fetchall()
        m_cols = [r[0] for r in m_cols_info]
        m_cols_lower = [c.lower() for c in m_cols]

        id_cols = [c for c, cl in zip(m_cols, m_cols_lower) if cl in ['global_id', 'fid', 'code', 'id', 'point_id']]
        
        if id_cols:
            where_clauses = [f"CAST({col} AS VARCHAR) = ?" for col in id_cols]
            sql = f"SELECT * FROM metadata WHERE {' OR '.join(where_clauses)} LIMIT 1"
            
            row = m_con.execute(sql, [target_id] * len(id_cols)).fetchone()
            if not row and clean_id != target_id:
                row = m_con.execute(sql, [clean_id] * len(id_cols)).fetchone()

            if row:
                metadata = dict(zip(m_cols_lower, row))
        m_con.close()
    except Exception as e:
        print(f"Metadata query error: {e}")

    # 2. Dynamic Time Series Match in Parquet
    try:
        ts_con = duckdb.connect()
        ts_cols_info = ts_con.execute("DESCRIBE SELECT * FROM read_parquet('timeseries.parquet')").fetchall()
        ts_cols = [r[0] for r in ts_cols_info]
        ts_cols_lower = [c.lower() for c in ts_cols]

        ts_id_cols = [c for c, cl in zip(ts_cols, ts_cols_lower) if cl in ['global_id', 'fid', 'code', 'id', 'point_id']]
        date_col = next((c for c, cl in zip(ts_cols, ts_cols_lower) if 'date' in cl or 'time' in cl or 'epoch' in cl), None)
        disp_col = next((c for c, cl in zip(ts_cols, ts_cols_lower) if 'disp' in cl or 'val' in cl or 'd' == cl), None)

        # Build candidate search list (request ID + metadata FID/CODE)
        search_vals = list(dict.fromkeys([
            target_id, 
            clean_id, 
            str(metadata.get('global_id', '')), 
            str(metadata.get('fid', '')), 
            str(metadata.get('code', ''))
        ]))
        search_vals = [v for v in search_vals if v]

        if ts_id_cols and date_col and disp_col:
            ts_rows = []
            for id_col in ts_id_cols:
                for val in search_vals:
                    query = f"""
                        SELECT CAST({date_col} AS VARCHAR) as d, CAST({disp_col} AS DOUBLE) as val
                        FROM read_parquet('timeseries.parquet')
                        WHERE CAST({id_col} AS VARCHAR) = ?
                        ORDER BY {date_col} ASC
                    """
                    res = ts_con.execute(query, [str(val)]).fetchall()
                    if res:
                        ts_rows = res
                        break
                if ts_rows:
                    break

            dates = [r[0] for r in ts_rows]
            displacement = [float(r[1]) if r[1] is not None else 0.0 for r in ts_rows]

        ts_con.close()
    except Exception as e:
        print(f"Time series query error: {e}")

    return {
        "metadata": metadata,
        "dates": dates,
        "displacement": displacement
    }