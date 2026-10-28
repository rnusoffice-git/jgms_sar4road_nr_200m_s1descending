import os
import json
import duckdb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
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
    """Pre-downloads metadata and reference points during startup."""
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
    Streams GeoJSON points in 5,000-row chunks directly to the browser.
    Eliminates high memory usage and avoids both 500 and 502 errors.
    """
    ensure_file_exists("metadata.duckdb")

    def generate_geojson_chunks():
        yield '{"type":"FeatureCollection","features":['

        con = duckdb.connect("metadata.duckdb", read_only=True)
        cursor = con.cursor()
        cursor.execute("SELECT global_id, latitude, longitude, vel FROM metadata")

        first_batch = True
        batch_size = 5000

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            chunk_items = []
            for r in rows:
                gid = str(r[0])
                lat = float(r[1]) if r[1] is not None else 0.0
                lon = float(r[2]) if r[2] is not None else 0.0
                vel = float(r[3]) if r[3] is not None else 0.0

                item = (
                    f'{{"type":"Feature","geometry":{{"type":"Point","coordinates":[{lon},{lat}]}},'
                    f'"properties":{{"global_id":"{gid}","velocity":{vel}}}}}'
                )
                chunk_items.append(item)

            prefix = "" if first_batch else ","
            yield prefix + ",".join(chunk_items)
            first_batch = False

        con.close()
        yield ']}'

    return StreamingResponse(generate_geojson_chunks(), media_type="application/json")

@app.get("/point/{global_id}")
def get_point_details(global_id: str):
    """Fetches metadata and time-series displacement data for a clicked point."""
    ensure_file_exists("metadata.duckdb")
    
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