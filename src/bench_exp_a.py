import os
import time
import psutil
import psycopg2
import numpy as np
import rasterio
from rasterio.windows import Window
import csv

DB_URI = os.getenv("DATABASE_URL", "postgresql://benchuser:benchpassword@db:5432/benchdb")
DATA_DIR = "/data/raw_rasters"
RESULTS_DIR = "/workspace/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_FILE = os.path.join(RESULTS_DIR, "exp_a_raster_diff.csv")

SCALES = {
    "1x": 512,
    "4x": 1024,
    "16x": 2048
}
TILE_SIZES = [32, 128, 256]
REPEATS = 3  # 每组重复跑 3 次取中位数

def get_db():
    conn = psycopg2.connect(DB_URI)
    conn.autocommit = True
    return conn

def get_wal_lsn(cur):
    cur.execute("SELECT pg_current_wal_lsn();")
    return cur.fetchone()[0]

def get_wal_diff_mb(cur, lsn_start, lsn_end):
    cur.execute("SELECT pg_wal_lsn_diff(%s, %s);", (lsn_end, lsn_start))
    return cur.fetchone()[0] / (1024 * 1024)

def get_table_size_mb(cur, table_name):
    cur.execute(f"SELECT pg_total_relation_size('{table_name}');")
    return cur.fetchone()[0] / (1024 * 1024)

# ============================================================
# 1. 方案一: Python (NumPy) 基准计算
# ============================================================
def bench_python(scale_name, size):
    win = Window(col_off=500, row_off=500, width=size, height=size)
    with rasterio.open(f"{DATA_DIR}/2023_B04_tokyo.tif") as s: b04_23 = s.read(1, window=win).astype(np.float32)
    with rasterio.open(f"{DATA_DIR}/2023_B08_tokyo.tif") as s: b08_23 = s.read(1, window=win).astype(np.float32)
    with rasterio.open(f"{DATA_DIR}/2024_B04_tokyo.tif") as s: b04_24 = s.read(1, window=win).astype(np.float32)
    with rasterio.open(f"{DATA_DIR}/2024_B08_tokyo.tif") as s: b08_24 = s.read(1, window=win).astype(np.float32)

    times = []
    mem_peaks = []
    proc = psutil.Process()

    for _ in range(REPEATS):
        mem_start = proc.memory_info().rss
        t0 = time.perf_counter()

        eps = 1e-6
        ndvi_23 = (b08_23 - b04_23) / (b08_23 + b04_23 + eps)
        ndvi_24 = (b08_24 - b04_24) / (b08_24 + b04_24 + eps)
        dndvi = ndvi_24 - ndvi_23
        mask = (dndvi < -0.2).astype(np.uint8)

        t1 = time.perf_counter()
        mem_end = proc.memory_info().rss
        times.append(t1 - t0)
        mem_peaks.append(max(0, (mem_end - mem_start) / (1024 * 1024)))

    return {
        "engine": "Python_NumPy",
        "scale": scale_name,
        "pixels": size * size,
        "tile_size": "N/A",
        "time_median": round(float(np.median(times)), 4),
        "wal_mb": 0.0,
        "disk_mb": round((dndvi.nbytes + mask.nbytes) / (1024 * 1024), 2),
        "mem_peak_mb": round(float(np.median(mem_peaks)), 2)
    }

# ============================================================
# 2. 方案二: PostGIS In-DB 计算
# ============================================================
def setup_indb_tables(conn, size, tile_size):
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS r_2023_indb; DROP TABLE IF EXISTS r_2024_indb; DROP TABLE IF EXISTS r_diff_indb;")
    cur.execute(f"""
        CREATE TABLE r_2023_indb (rid serial primary key, b4 raster, b8 raster);
        CREATE TABLE r_2024_indb (rid serial primary key, b4 raster, b8 raster);
    """)

    win = Window(col_off=500, row_off=500, width=size, height=size)
    with rasterio.open(f"{DATA_DIR}/2023_B04_tokyo.tif") as s: b04_23 = s.read(1, window=win).astype(np.float32)
    with rasterio.open(f"{DATA_DIR}/2023_B08_tokyo.tif") as s: b08_23 = s.read(1, window=win).astype(np.float32)
    with rasterio.open(f"{DATA_DIR}/2024_B04_tokyo.tif") as s: b04_24 = s.read(1, window=win).astype(np.float32)
    with rasterio.open(f"{DATA_DIR}/2024_B08_tokyo.tif") as s: b08_24 = s.read(1, window=win).astype(np.float32)

    # 切瓦片灌入库
    for y in range(0, size, tile_size):
        for x in range(0, size, tile_size):
            t_b4_23 = b04_23[y:y+tile_size, x:x+tile_size]
            t_b8_23 = b08_23[y:y+tile_size, x:x+tile_size]
            t_b4_24 = b04_24[y:y+tile_size, x:x+tile_size]
            t_b8_24 = b08_24[y:y+tile_size, x:x+tile_size]

            th, tw = t_b4_23.shape
            # 通过 ST_MakeEmptyRaster 构建瓦片
            cur.execute(f"""
                INSERT INTO r_2023_indb (b4, b8) VALUES (
                    ST_SetValues(ST_AddBand(ST_MakeEmptyRaster({tw}, {th}, {x}, {-y}, 1, -1, 0, 0, 0), '32BF'), 1, 1, 1, %s::double precision[][]),
                    ST_SetValues(ST_AddBand(ST_MakeEmptyRaster({tw}, {th}, {x}, {-y}, 1, -1, 0, 0, 0), '32BF'), 1, 1, 1, %s::double precision[][])
                );
                INSERT INTO r_2024_indb (b4, b8) VALUES (
                    ST_SetValues(ST_AddBand(ST_MakeEmptyRaster({tw}, {th}, {x}, {-y}, 1, -1, 0, 0, 0), '32BF'), 1, 1, 1, %s::double precision[][]),
                    ST_SetValues(ST_AddBand(ST_MakeEmptyRaster({tw}, {th}, {x}, {-y}, 1, -1, 0, 0, 0), '32BF'), 1, 1, 1, %s::double precision[][])
                );
            """, (t_b4_23.tolist(), t_b8_23.tolist(), t_b4_24.tolist(), t_b8_24.tolist()))

    cur.close()

def bench_postgis_indb(scale_name, size, tile_size):
    conn = get_db()
    cur = conn.cursor()
    setup_indb_tables(conn, size, tile_size)

    times, wals, disk_sizes = [], [], []

    for _ in range(REPEATS):
        cur.execute("DROP TABLE IF EXISTS r_diff_indb;")
        cur.execute("CHECKPOINT;")
        lsn_start = get_wal_lsn(cur)
        t0 = time.perf_counter()

        cur.execute("""
            CREATE TABLE r_diff_indb AS
            WITH 
            t23 AS (
                SELECT rid, ST_MapAlgebra(b8, 1, b4, 1, '(([rast1.val] - [rast2.val]) / ([rast1.val] + [rast2.val] + 0.000001))::float', '32BF') AS ndvi
                FROM r_2023_indb
            ),
            t24 AS (
                SELECT rid, ST_MapAlgebra(b8, 1, b4, 1, '(([rast1.val] - [rast2.val]) / ([rast1.val] + [rast2.val] + 0.000001))::float', '32BF') AS ndvi
                FROM r_2024_indb
            )
            SELECT t24.rid, ST_MapAlgebra(t24.ndvi, 1, t23.ndvi, 1, '([rast1.val] - [rast2.val])::float', '32BF') AS dndvi
            FROM t24 JOIN t23 ON t24.rid = t23.rid;
        """)

        t1 = time.perf_counter()
        lsn_end = get_wal_lsn(cur)

        times.append(t1 - t0)
        wals.append(get_wal_diff_mb(cur, lsn_start, lsn_end))
        disk_sizes.append(get_table_size_mb(cur, "r_diff_indb"))

    cur.execute("DROP TABLE IF EXISTS r_2023_indb; DROP TABLE IF EXISTS r_2024_indb; DROP TABLE IF EXISTS r_diff_indb;")
    cur.close()
    conn.close()

    return {
        "engine": "PostGIS_InDB",
        "scale": scale_name,
        "pixels": size * size,
        "tile_size": tile_size,
        "time_median": round(float(np.median(times)), 4),
        "wal_mb": round(float(np.median(wals)), 2),
        "disk_mb": round(float(np.median(disk_sizes)), 2),
        "mem_peak_mb": 0.0
    }

# ============================================================
# 主执行入口
# ============================================================
def main():
    print("=" * 75)
    print(">>> 启动 Experiment A: 栅格差值引擎评测 (PostGIS vs Python)")
    print("=" * 75)

    results = []

    for scale_name, size in SCALES.items():
        print(f"\n--- 测试规模: {scale_name} ({size}x{size} 像元, 共 {size*size:,} 像素) ---")
        
        # 1. Python 测试
        print(">> 运行 Python (NumPy)...")
        res_py = bench_python(scale_name, size)
        results.append(res_py)
        print(f"   耗时: {res_py['time_median']}s | 内存: {res_py['mem_peak_mb']}MB")

        # 2. PostGIS In-DB 测试不同瓦片大小
        for ts in TILE_SIZES:
            print(f">> 运行 PostGIS In-DB (瓦片大小: {ts}x{ts})...")
            res_pg = bench_postgis_indb(scale_name, size, ts)
            results.append(res_pg)
            print(f"   耗时: {res_pg['time_median']}s | WAL 暴涨: {res_pg['wal_mb']}MB | 表大小: {res_pg['disk_mb']}MB")

    # 写出 CSV
    keys = results[0].keys()
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 75)
    print(f"实验 A 完成！数据已存至: {CSV_FILE}")
    print("=" * 75)

if __name__ == "__main__":
    main()
