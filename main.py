import os
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from huggingface_hub import hf_hub_download
import duckdb

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
    Constructs GeoJSON directly in DuckDB's C++ engine to eliminate 
    Python dictionary allocation overhead and prevent 502 OOM crashes on Render.
    """
    con = duckdb.connect("metadata.duckdb", read_only=True)
    
    query = """
    SELECT '{"type":"FeatureCollection","features":[' || 
           string_agg(
               '{"type":"Feature","geometry":{"type":"Point","coordinates":[' || 
               CAST(longitude AS VARCHAR) || ',' || CAST(latitude AS VARCHAR) || 
               ']},"properties":{"global_id":"' || CAST(global_id AS VARCHAR) || 
               '","velocity":' || COALESCE(CAST(vel AS VARCHAR), '0.0') || '}}', 
               ','
           ) || ']}' 
    FROM metadata
    """
    
    json_str = con.execute(query).fetchone()[0]
    con.close()

    if not json_str:
        json_str = '{"type":"FeatureCollection","features":[]}'

    return Response(content=json_str, media_type="application/json")

@app.get("/point/{global_id}")
def get_point_details(global_id: str):
    """Fetches metadata and time-series displacement data for a clicked point."""
    meta_con = duckdb.connect("metadata.duckdb", read_only=True)
    meta_res = meta_con.execute("SELECT * FROM metadata WHERE global_id = ?", [global_id]).fetchone()
    meta_cols = [desc[0] for desc in meta_con.description] if meta_res else []
    metadata = dict(zip(meta_cols, meta_res)) if meta_res else {}
    meta_con.close()

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