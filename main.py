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
    """Pre-downloads essential metadata, reference points, and time-series parquet on startup."""
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

    if None not in (min_lat, max_lat, min_lon, max_lon):
        query = """
            SELECT global_id, latitude, longitude, vel 
            FROM metadata 
            WHERE latitude BETWEEN ? AND ? 
              AND longitude BETWEEN ? AND ?
            LIMIT ?
        """
        params = [min_lat, max_lat, min_lon, max_lon, limit]
    else:
        query = "SELECT global_id, latitude, longitude, vel FROM metadata LIMIT 3000"
        params = [limit]

    rows = con.execute(query, params).fetchall()
    con.close()

    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [r[2], r[1]]
            },
            "properties": {
                "global_id": str(r[0]),
                "velocity": float(r[3]) if r[3] is not None else 0.0
            }
        }
        for r in rows
    ]
    return {"type": "FeatureCollection", "features": features}

@app.get("/point/{global_id}")
def get_point_details(global_id: str):
    """Fetches metadata and time-series displacement data safely from local cached files."""
    ensure_file_exists("metadata.duckdb")
    ensure_file_exists("timeseries.parquet")

    metadata = {}
    dates = []
    displacement = []

    # 1. Fetch Metadata from DuckDB
    try:
        meta_con = duckdb.connect("metadata.duckdb", read_only=True)
        meta_res = meta_con.execute(
            "SELECT * FROM metadata WHERE CAST(global_id AS VARCHAR) = CAST(? AS VARCHAR)", 
            [str(global_id)]
        ).fetchone()

        if meta_res:
            cols = [desc[0].lower() for desc in meta_con.description]
            metadata = dict(zip(cols, meta_res))
        meta_con.close()
    except Exception as e:
        print(f"Error fetching metadata for global_id {global_id}: {e}")

    # 2. Fetch Time Series from Local Parquet File (Fast Columnar Scan)
    try:
        ts_con = duckdb.connect()
        ts_res = ts_con.execute("""
            SELECT CAST(date AS VARCHAR) as date_str, displacement 
            FROM read_parquet('timeseries.parquet') 
            WHERE CAST(global_id AS VARCHAR) = CAST(? AS VARCHAR)
            ORDER BY date ASC
        """, [str(global_id)]).fetchall()

        dates = [row[0] for row in ts_res]
        displacement = [float(row[1]) if row[1] is not None else 0.0 for row in ts_res]
        ts_con.close()
    except Exception as e:
        print(f"Error fetching time-series for global_id {global_id}: {e}")

    return {
        "metadata": metadata,
        "dates": dates,
        "displacement": displacement
    }