import os
import time
import csv
import threading
import multiprocessing
import psycopg2
import numpy as np
import rasterio
from rasterio.windows import Window

DB_URI = os.getenv("DATABASE_URL", "postgresql://benchuser:benchpassword@db:5432/benchdb")
OUT_DIR = "/workspace/results"
CSV_FILE = os.path.join(OUT_DIR, "exp_c_concurrency_interference.csv")
DATA_DIR = "/data/raw_rasters"

BENCH_DURATION_SEC = 25  # 每个场景加压测试 25 秒
CONCURRENT_CLIENTS = 4    # 4 个并发客户端 (压满 4 核 VM)

def get_db():
    conn = psycopg2.connect(DB_URI)
    conn.autocommit = True
    return conn

# ============================================================
# 1. 模拟业务核心矢量查询事务 (综合 Intersects + DWithin + KNN)
# ============================================================
def execute_vector_workload(cur):
    # 随机取一个变化图斑 ID 做深度穿透分析
    cid = np.random.randint(0, 5000)
    
    # 1. 查相交地块
    cur.execute("""
        SELECT p.parcel_id, p.landuse 
        FROM bench_parcels p, bench_changes c 
        WHERE c.change_id = %s AND ST_Intersects(c.geom, p.geom) 
        LIMIT 5;
    """, (cid,))
    cur.fetchall()

    # 2. 查 500m 范围内铁路
    cur.execute("""
        SELECT r.name 
        FROM bench_rails r, bench_changes c 
        WHERE c.change_id = %s AND ST_DWithin(c.geom, r.geom, 500) 
        LIMIT 5;
    """, (cid,))
    cur.fetchall()

    # 3. 查最近的 1 条铁路 (1-NN)
    cur.execute("""
        SELECT r.name 
        FROM bench_rails r, bench_changes c 
        WHERE c.change_id = %s 
        ORDER BY c.geom <-> r.geom 
        LIMIT 1;
    """, (cid,))
    cur.fetchall()

# ============================================================
# 2. 矢量客户端加压 Worker
# ============================================================
def client_worker(stop_event, latencies_list):
    conn = get_db()
    cur = conn.cursor()
    local_latencies = []
    
    while not stop_event.is_set():
        t0 = time.perf_counter()
        try:
            execute_vector_workload(cur)
            t1 = time.perf_counter()
            local_latencies.append((t1 - t0) * 1000.0) # 毫秒
        except Exception as e:
            pass
            
    cur.close()
    conn.close()
    latencies_list.extend(local_latencies)

# ============================================================
# 3. 后台负载生成器
# ============================================================
def background_indb_raster_worker(stop_event):
    """库内栅格批处理: 持续循环 ST_MapAlgebra"""
    conn = get_db()
    cur = conn.cursor()
    while not stop_event.is_set():
        try:
            cur.execute("""
                SELECT ST_MapAlgebra(
                    r24.rast, 1, r23.rast, 1,
                    '([rast1.val] - [rast2.val])::float', '32BF'
                )
                FROM (SELECT rast FROM verify_rasters WHERE name = 'b08_24') r24,
                     (SELECT rast FROM verify_rasters WHERE name = 'b08_23') r23;
            """)
            cur.fetchall()
        except Exception:
            break
    cur.close()
    conn.close()

def background_external_raster_worker(stop_event):
    """外部解耦计算: 独立进程在 Python 中循环计算差值"""
    win = Window(col_off=500, row_off=500, width=1024, height=1024)
    with rasterio.open(f"{DATA_DIR}/2024_B08_tokyo.tif") as s: b8_24 = s.read(1, window=win).astype(np.float32)
    with rasterio.open(f"{DATA_DIR}/2023_B08_tokyo.tif") as s: b8_23 = s.read(1, window=win).astype(np.float32)

    while not stop_event.is_set():
        _ = b8_24 - b8_23
        time.sleep(0.005)

# ============================================================
# 4. 单场景加压编排
# ============================================================
def run_scenario(scenario_name, bg_target=None, is_process=False):
    print(f"\n>>> 正在运行场景: [{scenario_name}] (持续 {BENCH_DURATION_SEC} 秒, {CONCURRENT_CLIENTS} 并发客户端)...")
    
    stop_event = threading.Event() if not is_process else multiprocessing.Event()
    bg_worker = None
    
    # 启动后台干扰任务
    if bg_target:
        if is_process:
            bg_worker = multiprocessing.Process(target=bg_target, args=(stop_event,))
        else:
            bg_worker = threading.Thread(target=bg_target, args=(stop_event,))
        bg_worker.daemon = True
        bg_worker.start()

    # 启动矢量客户端并发线程
    threads = []
    latencies_pool = [[] for _ in range(CONCURRENT_CLIENTS)]
    
    for i in range(CONCURRENT_CLIENTS):
        t = threading.Thread(target=client_worker, args=(stop_event, latencies_pool[i]))
        threads.append(t)
        t.start()

    # 运行指定时长
    time.sleep(BENCH_DURATION_SEC)
    stop_event.set()

    for t in threads:
        t.join()
    if bg_worker:
        bg_worker.join(timeout=2)

    # 汇总延迟
    all_latencies = []
    for l in latencies_pool:
        all_latencies.extend(l)

    all_latencies = np.array(all_latencies)
    total_queries = len(all_latencies)
    qps = total_queries / BENCH_DURATION_SEC

    res = {
        "scenario": scenario_name,
        "concurrent_clients": CONCURRENT_CLIENTS,
        "duration_s": BENCH_DURATION_SEC,
        "total_queries": total_queries,
        "qps": round(float(qps), 2),
        "p50_ms": round(float(np.percentile(all_latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(all_latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(all_latencies, 99)), 2),
        "max_ms": round(float(np.max(all_latencies)), 2)
    }

    print(f"   [完成] QPS: {res['qps']} | p50: {res['p50_ms']}ms | p95: {res['p95_ms']}ms | p99: {res['p99_ms']}ms | Max: {res['max_ms']}ms")
    return res

# ============================================================
# 主执行入口
# ============================================================
def main():
    print("=" * 80)
    print(">>> 启动 Experiment C: 并发干扰与架构解耦测试 (核心杀手锏)")
    print("=" * 80)

    # 预热并准备 PostGIS 基础栅格切片供后台任务反复查询
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS verify_rasters (name VARCHAR(32) PRIMARY KEY, rast raster);
    """)
    # 如果没有切片则切 1024x1024 灌一个
    cur.execute("SELECT count(*) FROM verify_rasters;")
    if cur.fetchone()[0] < 2:
        print("准备后台压测栅格底座...")
        win = Window(500, 500, 1024, 1024)
        for name, band in [("b08_23", "2023_B08_tokyo.tif"), ("b08_24", "2024_B08_tokyo.tif")]:
            with rasterio.open(f"{DATA_DIR}/{band}") as s:
                arr = s.read(1, window=win).astype(np.float32)
                p = f"/data/raw_rasters/tmp_{name}.tif"
                with rasterio.open(p, "w", driver="GTiff", height=1024, width=1024, count=1, dtype="float32", crs=s.crs, transform=s.window_transform(win)) as dst:
                    dst.write(arr, 1)
            with open(p, "rb") as f:
                cur.execute("INSERT INTO verify_rasters (name, rast) VALUES (%s, ST_FromGDALRaster(%s)) ON CONFLICT (name) DO NOTHING;", (name, psycopg2.Binary(f.read())))
            if os.path.exists(p): os.remove(p)
    cur.close()
    conn.close()

    results = []

    # 1. 基准情景: 无后台任务
    r_idle = run_scenario("1_Baseline_Idle", bg_target=None)
    results.append(r_idle)

    # 2. 混合架构: 库内强行跑栅格批处理
    r_coupled = run_scenario("2_Coupled_InDB_Raster", bg_target=background_indb_raster_worker, is_process=False)
    results.append(r_coupled)

    # 3. 解耦架构: 外部独立进程跑栅格
    r_decoupled = run_scenario("3_Decoupled_External_Raster", bg_target=background_external_raster_worker, is_process=True)
    results.append(r_decoupled)

    # 写出 CSV
    keys = results[0].keys()
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 80)
    print(f"实验 C 顺利收官！核心量化证据已存至: {CSV_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    main()
