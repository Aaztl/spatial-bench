import os
import time
import io
import csv
import psycopg2
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.wkb import dumps

DB_URI = os.getenv("DATABASE_URL", "postgresql://benchuser:benchpassword@db:5432/benchdb")
DATA_DIR = "/data/raw_rasters"
OUT_DIR = "/workspace/results"
CSV_FILE = os.path.join(OUT_DIR, "exp_d_end_to_end.csv")

# 统一测试规模：1024x1024 像元 (约 105 万像素)
TEST_SIZE = 1024
CRS_EPSG = 32654

def get_db():
    conn = psycopg2.connect(DB_URI)
    conn.autocommit = True
    return conn

# ============================================================
# 流水线 1: 解耦混合架构 (Python Compute/Vectorize -> COPY -> PostGIS)
# ============================================================
def run_pipeline_decoupled():
    print("\n" + "=" * 60)
    print(">>> 运行流水线 1: 解耦架构 (Python 计算/矢量化 -> COPY 入库 -> PostGIS 分析)")
    print("=" * 60)
    
    stages = {}
    total_start = time.perf_counter()

    # 1. 栅格计算 (Compute)
    t0 = time.perf_counter()
    win = Window(500, 500, TEST_SIZE, TEST_SIZE)
    with rasterio.open(f"{DATA_DIR}/2023_B04_tokyo.tif") as s: b4_23 = s.read(1, window=win).astype(np.float32)
    with rasterio.open(f"{DATA_DIR}/2023_B08_tokyo.tif") as s: b8_23 = s.read(1, window=win).astype(np.float32)
    with rasterio.open(f"{DATA_DIR}/2024_B04_tokyo.tif") as s: b4_24 = s.read(1, window=win).astype(np.float32)
    with rasterio.open(f"{DATA_DIR}/2024_B08_tokyo.tif") as s: 
        b8_24 = s.read(1, window=win).astype(np.float32)
        transform = s.window_transform(win)

    eps = 1e-6
    ndvi_23 = (b8_23 - b4_23) / (b8_23 + b4_23 + eps)
    ndvi_24 = (b8_24 - b4_24) / (b8_24 + b4_24 + eps)
    dndvi = ndvi_24 - ndvi_23
    change_mask = (dndvi < -0.15).astype(np.uint8)
    t1 = time.perf_counter()
    stages["1_Compute_NDVI_s"] = t1 - t0
    print(f" [Stage 1] Python 差值计算耗时: {stages['1_Compute_NDVI_s']:.4f}s")

    # 2. 多边形化 (Vectorization)
    t0 = time.perf_counter()
    gen = shapes(change_mask, mask=(change_mask == 1), transform=transform)
    poly_records = []
    for geom_dict, val in gen:
        poly_records.append(dumps(shape(geom_dict), hex=True, srid=CRS_EPSG))
    t1 = time.perf_counter()
    stages["2_Polygonize_s"] = t1 - t0
    n_polys = len(poly_records)
    print(f" [Stage 2] Python 矢量化耗时: {stages['2_Polygonize_s']:.4f}s (生成变化多边形: {n_polys:,} 个)")

    # 3. 高速流式入库 (COPY + GiST Index)
    t0 = time.perf_counter()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS e2e_decoupled_changes CASCADE;")
    cur.execute("CREATE TABLE e2e_decoupled_changes (id serial, geom geometry(Polygon, 32654));")

    sio = io.StringIO()
    for wkb in poly_records:
        sio.write(f"{wkb}\n")
    sio.seek(0)
    cur.copy_from(sio, "e2e_decoupled_changes", columns=("geom",))
    
    cur.execute("CREATE INDEX idx_e2e_dec_geom ON e2e_decoupled_changes USING GIST(geom);")
    cur.execute("ANALYZE e2e_decoupled_changes;")
    t1 = time.perf_counter()
    stages["3_Ingest_COPY_Index_s"] = t1 - t0
    throughput = n_polys / (stages["3_Ingest_COPY_Index_s"] + 1e-6)
    print(f" [Stage 3] COPY 入库与建索引耗时: {stages['3_Ingest_COPY_Index_s']:.4f}s (吞吐率: {throughput:,.0f} poly/s)")

    # 4. 空间分析关联 (Spatial Query: 与地块相交)
    t0 = time.perf_counter()
    cur.execute("""
        SELECT count(*), count(DISTINCT p.parcel_id)
        FROM e2e_decoupled_changes c
        JOIN bench_parcels p ON ST_Intersects(c.geom, p.geom);
    """)
    res = cur.fetchall()
    t1 = time.perf_counter()
    stages["4_Spatial_Analysis_s"] = t1 - t0
    print(f" [Stage 4] PostGIS 碰撞分析耗时: {stages['4_Spatial_Analysis_s']:.4f}s (碰撞对数: {res[0][0]:,})")

    total_time = time.perf_counter() - total_start
    stages["Total_Wall_Time_s"] = total_time
    cur.close()
    conn.close()

    return {
        "pipeline": "Decoupled_Hybrid",
        "polygons_detected": n_polys,
        **stages
    }

# ============================================================
# 流水线 2: 纯库内架构 (All-in-DB: Ingest Raw -> ST_MapAlgebra -> ST_DumpAsPolygons)
# ============================================================
def run_pipeline_all_indb():
    print("\n" + "=" * 60)
    print(">>> 运行流水线 2: 纯库内架构 (Raw 灌库 -> ST_MapAlgebra -> ST_DumpAsPolygons)")
    print("=" * 60)

    stages = {}
    total_start = time.perf_counter()
    conn = get_db()
    cur = conn.cursor()

    # 1. 原始波段切片入库 (Ingest Raw Rasters 128x128 瓦片)
    t0 = time.perf_counter()
    cur.execute("DROP TABLE IF EXISTS e2e_indb_raw CASCADE; DROP TABLE IF EXISTS e2e_indb_diff CASCADE; DROP TABLE IF EXISTS e2e_indb_changes CASCADE;")
    cur.execute("CREATE TABLE e2e_indb_raw (rid serial primary key, b4_23 raster, b8_23 raster, b4_24 raster, b8_24 raster);")

    win = Window(500, 500, TEST_SIZE, TEST_SIZE)
    with rasterio.open(f"{DATA_DIR}/2023_B04_tokyo.tif") as s: b4_23 = s.read(1, window=win).astype(np.float32)
    with rasterio.open(f"{DATA_DIR}/2023_B08_tokyo.tif") as s: b8_23 = s.read(1, window=win).astype(np.float32)
    with rasterio.open(f"{DATA_DIR}/2024_B04_tokyo.tif") as s: b4_24 = s.read(1, window=win).astype(np.float32)
    with rasterio.open(f"{DATA_DIR}/2024_B08_tokyo.tif") as s: b8_24 = s.read(1, window=win).astype(np.float32)

    tile_size = 128
    for y in range(0, TEST_SIZE, tile_size):
        for x in range(0, TEST_SIZE, tile_size):
            t_b4_23 = b4_23[y:y+tile_size, x:x+tile_size]
            t_b8_23 = b8_23[y:y+tile_size, x:x+tile_size]
            t_b4_24 = b4_24[y:y+tile_size, x:x+tile_size]
            t_b8_24 = b8_24[y:y+tile_size, x:x+tile_size]
            th, tw = t_b4_23.shape

            cur.execute(f"""
                INSERT INTO e2e_indb_raw (b4_23, b8_23, b4_24, b8_24) VALUES (
                    ST_SetValues(ST_AddBand(ST_MakeEmptyRaster({tw}, {th}, {x}, {-y}, 10, -10, 0, 0, 32654), '32BF'), 1, 1, 1, %s::double precision[][]),
                    ST_SetValues(ST_AddBand(ST_MakeEmptyRaster({tw}, {th}, {x}, {-y}, 10, -10, 0, 0, 32654), '32BF'), 1, 1, 1, %s::double precision[][]),
                    ST_SetValues(ST_AddBand(ST_MakeEmptyRaster({tw}, {th}, {x}, {-y}, 10, -10, 0, 0, 32654), '32BF'), 1, 1, 1, %s::double precision[][]),
                    ST_SetValues(ST_AddBand(ST_MakeEmptyRaster({tw}, {th}, {x}, {-y}, 10, -10, 0, 0, 32654), '32BF'), 1, 1, 1, %s::double precision[][])
                );
            """, (t_b4_23.tolist(), t_b8_23.tolist(), t_b4_24.tolist(), t_b8_24.tolist()))

    t1 = time.perf_counter()
    stages["1_Ingest_Raw_s"] = t1 - t0
    print(f" [Stage 1] 原始瓦片入库耗时: {stages['1_Ingest_Raw_s']:.4f}s")

    # 2. 库内栅格差值计算 (ST_MapAlgebra)
    t0 = time.perf_counter()
    cur.execute("""
        CREATE TABLE e2e_indb_diff AS
        WITH
        t23 AS (SELECT rid, ST_MapAlgebra(b8_23, 1, b4_23, 1, '(([rast1.val]-[rast2.val])/([rast1.val]+[rast2.val]+0.000001))::float', '32BF') AS ndvi FROM e2e_indb_raw),
        t24 AS (SELECT rid, ST_MapAlgebra(b8_24, 1, b4_24, 1, '(([rast1.val]-[rast2.val])/([rast1.val]+[rast2.val]+0.000001))::float', '32BF') AS ndvi FROM e2e_indb_raw)
        SELECT t24.rid, ST_MapAlgebra(t24.ndvi, 1, t23.ndvi, 1, 'CASE WHEN ([rast1.val]-[rast2.val]) < -0.15 THEN 1 ELSE 0 END', '8BUI') AS mask
        FROM t24 JOIN t23 ON t24.rid = t23.rid;
    """)
    t1 = time.perf_counter()
    stages["2_InDB_MapAlgebra_s"] = t1 - t0
    print(f" [Stage 2] 库内 ST_MapAlgebra 差值耗时: {stages['2_InDB_MapAlgebra_s']:.4f}s")

    # 3. 库内多边形化 (ST_DumpAsPolygons) - 语句严格拆分
    t0 = time.perf_counter()
    cur.execute("""
        CREATE TABLE e2e_indb_changes AS
        SELECT (gv).geom::geometry(Polygon, 32654) AS geom
        FROM (
            SELECT ST_DumpAsPolygons(mask) AS gv 
            FROM e2e_indb_diff
        ) sub
        WHERE (gv).val = 1;
    """)
    cur.execute("CREATE INDEX idx_e2e_indb_geom ON e2e_indb_changes USING GIST(geom);")
    cur.execute("ANALYZE e2e_indb_changes;")
    
    cur.execute("SELECT count(*) FROM e2e_indb_changes;")
    n_polys = cur.fetchone()[0]
    t1 = time.perf_counter()
    stages["3_InDB_DumpPolygons_Index_s"] = t1 - t0
    print(f" [Stage 3] 库内 ST_DumpAsPolygons 矢量化与索引耗时: {stages['3_InDB_DumpPolygons_Index_s']:.4f}s (多边形: {n_polys:,} 个)")

    # 4. 空间分析关联 (Spatial Query: 与地块相交)
    t0 = time.perf_counter()
    cur.execute("""
        SELECT count(*), count(DISTINCT p.parcel_id)
        FROM e2e_indb_changes c
        JOIN bench_parcels p ON ST_Intersects(c.geom, p.geom);
    """)
    res = cur.fetchall()
    t1 = time.perf_counter()
    stages["4_Spatial_Analysis_s"] = t1 - t0
    print(f" [Stage 4] PostGIS 碰撞分析耗时: {stages['4_Spatial_Analysis_s']:.4f}s (碰撞对数: {res[0][0]:,})")

    total_time = time.perf_counter() - total_start
    stages["Total_Wall_Time_s"] = total_time

    cur.close()
    conn.close()

    return {
        "pipeline": "All_In_DB",
        "polygons_detected": n_polys,
        **stages
    }

# ============================================================
# 主执行入口
# ============================================================
def main():
    print("=" * 80)
    print(">>> 启动 Experiment D: 端到端全链路流水线评测 (混合架构决战)")
    print("=" * 80)

    res_dec = run_pipeline_decoupled()
    res_indb = run_pipeline_all_indb()

    results = [res_dec, res_indb]
    
    print("\n" + "=" * 80)
    print(">>> 实验 D: 端到端耗时对比总榜")
    print("=" * 80)
    print(f"1. 解耦混合架构总耗时: {res_dec['Total_Wall_Time_s']:.2f}s  (提取图斑: {res_dec['polygons_detected']:,} 个)")
    print(f"2. 纯库内全流程总耗时: {res_indb['Total_Wall_Time_s']:.2f}s  (提取图斑: {res_indb['polygons_detected']:,} 个)")
    speedup = res_indb['Total_Wall_Time_s'] / res_dec['Total_Wall_Time_s']
    print(f"\n>>> 🚀 解耦混合架构全链路综合加速比: {speedup:.2f}x 倍！")
    print("=" * 80)

    keys = set()
    for r in results: keys.update(r.keys())
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(list(keys)))
        writer.writeheader()
        writer.writerows(results)
    print(f"结果已记录至: {CSV_FILE}")

if __name__ == "__main__":
    main()
