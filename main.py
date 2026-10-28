import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from huggingface_hub import hf_hub_download
import duckdb

app = FastAPI()

# Enable CORS for GitHub Pages frontend access
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
    """Downloads required dataset files from private Hugging Face repo if not cached locally."""
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
    """Pre-downloads lightweight metadata and reference points during startup."""
    print("Pre-downloading essential datasets...")
    ensure_file_exists("metadata.duckdb")
    ensure_file_exists("reference_points.geojson")
    print("Backend ready!")

@app.get("/")
def root():
    return {"status": "sar4road_dashboard Private Hugging Face API Online"}

@app.get("/geojson/reference_points.geojson")
def get_ref_points():
    ensure_file_exists("reference_points.geojson")
    return FileResponse("reference_points.geojson")

@app.get("/geojson/insar_map_points.geojson")
def get_insar_map_points():
    """
    Dynamically generates map point GeoJSON from metadata.duckdb on the fly.
    Bypasses downloading/loading the 935 MB GeoJSON file to stay within Render's 512 MB RAM limit.
    """
    con = duckdb.connect("metadata.duckdb", read_only=True)
    rows = con.execute("SELECT global_id, latitude, longitude, velocity FROM metadata").fetchall()
    con.close()

    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [r[2], r[1]]  # [longitude, latitude]
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
    """Fetches metadata and time-series displacement data for a clicked point."""
    # 1. Fetch point metadata from local DuckDB
    meta_con = duckdb.connect("metadata.duckdb", read_only=True)
    meta_res = meta_con.execute("SELECT * FROM metadata WHERE global_id = ?", [global_id]).fetchone()
    meta_cols = [desc[0] for desc in meta_con.description] if meta_res else []
    metadata = dict(zip(meta_cols, meta_res)) if meta_res else {}
    meta_con.close()

    # 2. Query time-series directly from remote Parquet on Hugging Face using DuckDB httpfs
    ts_con = duckdb.connect()
    ts_con.execute(f"""
        INSTALL httpfs; LOAD httpfs;
        SET http_headers = map(['Authorization'], ['Bearer {HF_TOKEN}']);
    """)

    parquet_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/timeseries.parquet"
    ts_res = ts_con.execute(f"""
        SELECT strftime(date, '%Y-%m-%d') as date, displacement 
        FROM read_parquet('{parquet_url}') 
        WHERE global_id = ? 
        ORDER BY date ASC
    """, [global_id]).fetchall()
    ts_con.close()

    return {
        "metadata": metadata,
        "dates": [row[0] for row in ts_res],
        "displacement": [row[1] for row in ts_res]
    }