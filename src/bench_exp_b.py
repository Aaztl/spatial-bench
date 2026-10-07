import os
import time
import csv
import psycopg2
from psycopg2.extras import execute_values
import duckdb
import geopandas as gpd
import numpy as np
from shapely.geometry import box

DB_URI = os.getenv("DATABASE_URL", "postgresql://benchuser:benchpassword@db:5432/benchdb")
OUT_DIR = "/workspace/results"
CSV_FILE = os.path.join(OUT_DIR, "exp_b_vector_queries.csv")
PARCELS_PATH = "/data/processed/base_parcels_100k.geojson"
RAILS_PATH = "/data/processed/tokyo_railways.geojson"

REPEATS = 3
TARGET_CRS = "EPSG:32654"  # UTM Zone 54N (米制坐标系)

def get_db():
    conn = psycopg2.connect(DB_URI)
    conn.autocommit = True
    return conn

# ============================================================
# 0. 准备与投影转换数据 (统一转为 UTM 54N)
# ============================================================
print(">>> 加载并重投影矢量底座至 EPSG:32654 (UTM Zone 54N)...")
gdf_parcels_all = gpd.read_file(PARCELS_PATH).to_crs(TARGET_CRS)
gdf_rails = gpd.read_file(RAILS_PATH).to_crs(TARGET_CRS)

# 生成 5,000 个模拟变化图斑
np.random.seed(99)
minx, miny, maxx, maxy = gdf_parcels_all.total_bounds
cx = np.random.uniform(minx, maxx, 5000)
cy = np.random.uniform(miny, maxy, 5000)
change_boxes = [box(cx[i], cy[i], cx[i]+50, cy[i]+50) for i in range(5000)]
gdf_changes = gpd.GeoDataFrame({"change_id": range(5000), "geometry": change_boxes}, crs=TARGET_CRS)

print(f"数据就绪: 地块={len(gdf_parcels_all)}, 铁路={len(gdf_rails)}, 变化图斑={len(gdf_changes)}")

# ============================================================
# 1. PostGIS 评测模块
# ============================================================
def setup_postgis_tables(conn, n_parcels):
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS bench_parcels CASCADE; DROP TABLE IF EXISTS bench_rails CASCADE; DROP TABLE IF EXISTS bench_changes CASCADE;")
    
    cur.execute("CREATE TABLE bench_parcels (parcel_id int, landuse text, geom geometry(Polygon, 32654));")
    cur.execute("CREATE TABLE bench_rails (name text, geom geometry(LineString, 32654));")
    cur.execute("CREATE TABLE bench_changes (change_id int, geom geometry(Polygon, 32654));")
    
    sub_parcels = gdf_parcels_all.iloc[:n_parcels]

    # 1. 批量入库 parcels
    parcel_data = [(p.parcel_id, p.landuse, p.geometry.wkb_hex) for p in sub_parcels.itertuples()]
    execute_values(cur, 
        "INSERT INTO bench_parcels (parcel_id, landuse, geom) VALUES %s",
        parcel_data,
        template="(%s, %s, ST_GeomFromWKB(decode(%s, 'hex'), 32654))",
        page_size=5000
    )

    # 2. 批量入库 rails
    rail_data = [(r.name, r.geometry.wkb_hex) for r in gdf_rails.itertuples()]
    execute_values(cur,
        "INSERT INTO bench_rails (name, geom) VALUES %s",
        rail_data,
        template="(%s, ST_GeomFromWKB(decode(%s, 'hex'), 32654))",
        page_size=5000
    )

    # 3. 批量入库 changes
    change_data = [(c.change_id, c.geometry.wkb_hex) for c in gdf_changes.itertuples()]
    execute_values(cur,
        "INSERT INTO bench_changes (change_id, geom) VALUES %s",
        change_data,
        template="(%s, ST_GeomFromWKB(decode(%s, 'hex'), 32654))",
        page_size=5000
    )
    
    # 分开执行，杜绝事务块报错
    cur.execute("VACUUM ANALYZE bench_parcels;")
    cur.execute("VACUUM ANALYZE bench_rails;")
    cur.execute("VACUUM ANALYZE bench_changes;")
    cur.close()

def bench_postgis(n_parcels, with_index=True):
    conn = get_db()
    cur = conn.cursor()
    setup_postgis_tables(conn, n_parcels)
    
    engine_name = "PostGIS_GiST" if with_index else "PostGIS_NoIndex"
    if with_index:
        cur.execute("CREATE INDEX idx_parcels_geom ON bench_parcels USING GIST(geom);")
        cur.execute("CREATE INDEX idx_rails_geom ON bench_rails USING GIST(geom);")
        cur.execute("CREATE INDEX idx_changes_geom ON bench_changes USING GIST(geom);")
        cur.execute("ANALYZE bench_parcels;")
        cur.execute("ANALYZE bench_rails;")
        cur.execute("ANALYZE bench_changes;")

    # Q1: 空间相交 (5000 变化 vs n 地块)
    q1 = "SELECT count(*) FROM bench_changes c JOIN bench_parcels p ON ST_Intersects(c.geom, p.geom);"
    times_q1 = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        cur.execute(q1)
        cur.fetchall()
        times_q1.append(time.perf_counter() - t0)

    # Q2: 500米范围查询 (5000 变化 vs 8148 铁路)
    q2 = "SELECT count(DISTINCT c.change_id) FROM bench_changes c JOIN bench_rails r ON ST_DWithin(c.geom, r.geom, 500);"
    times_q2 = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        cur.execute(q2)
        cur.fetchall()
        times_q2.append(time.perf_counter() - t0)

    # Q3: 1-NN 最近邻查询 (前 500 个变化图斑寻找最近的 1 条铁路)
    q3 = """
        SELECT c.change_id, r.name 
        FROM (SELECT * FROM bench_changes WHERE change_id < 500) c
        CROSS JOIN LATERAL (
            SELECT name FROM bench_rails r ORDER BY c.geom <-> r.geom LIMIT 1
        ) r;
    """
    times_q3 = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        cur.execute(q3)
        cur.fetchall()
        times_q3.append(time.perf_counter() - t0)

    cur.close()
    conn.close()

    return {
        "engine": engine_name,
        "n_parcels": n_parcels,
        "q1_intersects_s": round(float(np.median(times_q1)), 4),
        "q2_dwithin_s": round(float(np.median(times_q2)), 4),
        "q3_knn_s": round(float(np.median(times_q3)), 4)
    }

# ============================================================
# 2. GeoPandas 评测模块
# ============================================================
def bench_geopandas(n_parcels):
    sub_parcels = gdf_parcels_all.iloc[:n_parcels]

    # Q1: 空间相交 (5000 变化 vs n 地块)
    times_q1 = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        res1 = gpd.sjoin(gdf_changes, sub_parcels, how="inner", predicate="intersects")
        times_q1.append(time.perf_counter() - t0)

    # Q2: 500米范围查询 (5000 变化 vs 8148 铁路)
    times_q2 = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        buffered_rails = gdf_rails.copy()
        buffered_rails["geometry"] = buffered_rails.geometry.buffer(500)
        res2 = gpd.sjoin(gdf_changes, buffered_rails, how="inner", predicate="intersects")
        times_q2.append(time.perf_counter() - t0)

    # Q3: 1-NN 最近邻 (对铁路构建 STRtree，为前 500 个变化图斑寻找最近的 1 条铁路)
    sample_changes = gdf_changes.iloc[:500]
    times_q3 = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        res3 = gdf_rails.sindex.nearest(sample_changes.geometry, return_distance=False)
        times_q3.append(time.perf_counter() - t0)

    return {
        "engine": "GeoPandas_STRtree",
        "n_parcels": n_parcels,
        "q1_intersects_s": round(float(np.median(times_q1)), 4),
        "q2_dwithin_s": round(float(np.median(times_q2)), 4),
        "q3_knn_s": round(float(np.median(times_q3)), 4)
    }

# ============================================================
# 3. DuckDB Spatial 评测模块
# ============================================================
def bench_duckdb(n_parcels):
    con = duckdb.connect(":memory:")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("PRAGMA max_memory = '1.5GB';")
    con.execute("PRAGMA threads = 4;")
    
    sub_parcels = gdf_parcels_all.iloc[:n_parcels]

    # WKB 桥接
    p_df = sub_parcels[["parcel_id"]].copy()
    p_df["wkb"] = sub_parcels.geometry.to_wkb()

    r_df = gdf_rails[["name"]].copy()
    r_df["wkb"] = gdf_rails.geometry.to_wkb()

    c_df = gdf_changes[["change_id"]].copy()
    c_df["wkb"] = gdf_changes.geometry.to_wkb()

    con.register("p_view", p_df)
    con.register("r_view", r_df)
    con.register("c_view", c_df)
    
    con.execute("CREATE TABLE parcels AS SELECT parcel_id, ST_GeomFromWKB(wkb) AS geom FROM p_view;")
    con.execute("CREATE TABLE rails AS SELECT name, ST_GeomFromWKB(wkb) AS geom FROM r_view;")
    con.execute("CREATE TABLE changes AS SELECT change_id, ST_GeomFromWKB(wkb) AS geom FROM c_view;")

    # Q1: 空间相交
    times_q1 = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        con.execute("SELECT count(*) FROM changes c JOIN parcels p ON ST_Intersects(c.geom, p.geom);").fetchall()
        times_q1.append(time.perf_counter() - t0)

    # Q2: 500米范围查询
    times_q2 = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        con.execute("SELECT count(DISTINCT c.change_id) FROM changes c JOIN rails r ON ST_DWithin(c.geom, r.geom, 500);").fetchall()
        times_q2.append(time.perf_counter() - t0)

    # Q3: 1-NN 最近邻 (纯实测前 500 个变化图斑，无任何 * 10 折算)
    times_q3 = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        con.execute("""
            SELECT change_id, name FROM (
                SELECT c.change_id, r.name,
                       ROW_NUMBER() OVER (PARTITION BY c.change_id ORDER BY ST_Distance(c.geom, r.geom)) as rn
                FROM (SELECT * FROM changes WHERE change_id < 500) c CROSS JOIN rails r
            ) WHERE rn = 1;
        """).fetchall()
        times_q3.append(time.perf_counter() - t0)

    con.close()
    return {
        "engine": "DuckDB_Spatial",
        "n_parcels": n_parcels,
        "q1_intersects_s": round(float(np.median(times_q1)), 4),
        "q2_dwithin_s": round(float(np.median(times_q2)), 4),
        "q3_knn_s": round(float(np.median(times_q3)), 4)
    }

# ============================================================
# 主执行入口
# ============================================================
def main():
    print("=" * 80)
    print(">>> 启动 Experiment B: 矢量空间关联压测 (PostGIS vs GeoPandas vs DuckDB)")
    print("=" * 80)

    scales = [10_000, 50_000, 100_000]
    results = []

    for n in scales:
        print(f"\n--- 测试地块规模: {n:,} 个多边形 ---")
        
        # 1. GeoPandas
        print(">> 运行 GeoPandas (STRtree)...")
        r_gpd = bench_geopandas(n)
        results.append(r_gpd)
        print(f"   Intersects: {r_gpd['q1_intersects_s']}s | DWithin: {r_gpd['q2_dwithin_s']}s | KNN(1-NN): {r_gpd['q3_knn_s']}s")

        # 2. DuckDB Spatial
        print(">> 运行 DuckDB Spatial...")
        r_duck = bench_duckdb(n)
        results.append(r_duck)
        print(f"   Intersects: {r_duck['q1_intersects_s']}s | DWithin: {r_duck['q2_dwithin_s']}s | KNN(1-NN): {r_duck['q3_knn_s']}s")

        # 3. PostGIS GiST
        print(">> 运行 PostGIS (带 GiST 索引)...")
        r_pg_idx = bench_postgis(n, with_index=True)
        results.append(r_pg_idx)
        print(f"   Intersects: {r_pg_idx['q1_intersects_s']}s | DWithin: {r_pg_idx['q2_dwithin_s']}s | KNN(1-NN): {r_pg_idx['q3_knn_s']}s")

        # 4. PostGIS No-Index (仅在 10,000 规模下测一次作为对照组基线)
        if n == 10_000:
            print(">> 运行 PostGIS (无索引全表扫描基线)...")
            r_pg_noidx = bench_postgis(n, with_index=False)
            results.append(r_pg_noidx)
            print(f"   Intersects: {r_pg_noidx['q1_intersects_s']}s | DWithin: {r_pg_noidx['q2_dwithin_s']}s | KNN(1-NN): {r_pg_noidx['q3_knn_s']}s")

    # 写出 CSV
    keys = results[0].keys()
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 80)
    print(f"实验 B 完成！纯实测数据已存至: {CSV_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    main()
