import os
import urllib.request
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import duckdb

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
HF_DATASET_URL = f"https://huggingface.co/datasets/{HF_USERNAME}/sar4road-data/resolve/main"

def ensure_file_exists(filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename} from Hugging Face...")
        url = f"{HF_DATASET_URL}/{filename}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
        with urllib.request.urlopen(req) as response, open(filename, "wb") as out_file:
            out_file.write(response.read())
        print(f"{filename} download complete!")

@app.on_event("startup")
def startup_event():
    ensure_file_exists("metadata.duckdb")

@app.get("/")
def root():
    return {"status": "sar4road_dashboard Private Hugging Face API Online"}

@app.get("/geojson/{filename}")
def get_geojson(filename: str):
    ensure_file_exists(filename)
    return FileResponse(filename)

@app.get("/point/{global_id}")
def get_point_details(global_id: str):
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

    parquet_url = f"{HF_DATASET_URL}/timeseries.parquet"
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