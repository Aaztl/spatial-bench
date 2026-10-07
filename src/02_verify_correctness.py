import os
import numpy as np
import rasterio
from rasterio.windows import Window
import psycopg2
from psycopg2 import sql

DB_URI = os.getenv("DATABASE_URL", "postgresql://benchuser:benchpassword@db:5432/benchdb")
DATA_DIR = "/data/raw_rasters"
PATCH_SIZE = 512

def run_verification():
    print("=" * 60)
    print(">>> 启动 Phase 1: 像元级算子正确性验证 (Python vs PostGIS)")
    print("=" * 60)

    # 1. 读取 512x512 切片
    # 从中心位置偏移 (1000, 1000) 处切一块包含复杂地表的区域
    win = Window(col_off=1000, row_off=1000, width=PATCH_SIZE, height=PATCH_SIZE)

    with rasterio.open(f"{DATA_DIR}/2023_B04_tokyo.tif") as s:
        b04_23 = s.read(1, window=win).astype(np.float32)
        transform = s.window_transform(win)
        crs = s.crs
    with rasterio.open(f"{DATA_DIR}/2023_B08_tokyo.tif") as s:
        b08_23 = s.read(1, window=win).astype(np.float32)

    with rasterio.open(f"{DATA_DIR}/2024_B04_tokyo.tif") as s:
        b04_24 = s.read(1, window=win).astype(np.float32)
    with rasterio.open(f"{DATA_DIR}/2024_B08_tokyo.tif") as s:
        b08_24 = s.read(1, window=win).astype(np.float32)

    # 2. Python (NumPy) 基准计算
    eps = 1e-6
    ndvi_23_py = (b08_23 - b04_23) / (b08_23 + b04_23 + eps)
    ndvi_24_py = (b08_24 - b04_24) / (b08_24 + b04_24 + eps)
    dndvi_py = ndvi_24_py - ndvi_23_py
    print(f"[Python] NumPy 计算完成，样本像元数: {dndvi_py.size}")

    # 临时写出 4 个切片供 PostGIS 读取
    patch_paths = {}
    for name, arr in [("b04_23", b04_23), ("b08_23", b08_23), 
                      ("b04_24", b04_24), ("b08_24", b08_24)]:
        p_path = f"/data/raw_rasters/tmp_{name}.tif"
        patch_paths[name] = p_path
        with rasterio.open(
            p_path, "w", driver="GTiff",
            height=PATCH_SIZE, width=PATCH_SIZE, count=1,
            dtype="float32", crs=crs, transform=transform
        ) as dst:
            dst.write(arr, 1)

    # 3. 连接 PostGIS 并初始化环境
    conn = psycopg2.connect(DB_URI)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis_raster;")
    cur.execute("DROP TABLE IF EXISTS verify_rasters;")
    cur.execute("""
        CREATE TABLE verify_rasters (
            name VARCHAR(32) PRIMARY KEY,
            rast raster
        );
    """)

    # 4. 将切片加载入库 (使用 PostGIS 原生 ST_FromGDALRaster / WKB)
    for name, p_path in patch_paths.items():
        with open(p_path, "rb") as f:
            raw_bytes = f.read()
        cur.execute(
            "INSERT INTO verify_rasters (name, rast) VALUES (%s, ST_FromGDALRaster(%s));",
            (name, psycopg2.Binary(raw_bytes))
        )
    print("[PostGIS] 4 个切片已通过 ST_FromGDALRaster 加载进库。")

    # 5. PostGIS ST_MapAlgebra 计算
    # 表达式必须与 NumPy 的加 eps 逻辑完全一致
    mapalgebra_sql = """
    WITH 
    t2023 AS (
        SELECT ST_MapAlgebra(
            r8.rast, 1, r4.rast, 1,
            '(([rast1.val] - [rast2.val]) / ([rast1.val] + [rast2.val] + 0.000001))::float',
            '32BF'
        ) AS rast
        FROM verify_rasters r8, verify_rasters r4
        WHERE r8.name = 'b08_23' AND r4.name = 'b04_23'
    ),
    t2024 AS (
        SELECT ST_MapAlgebra(
            r8.rast, 1, r4.rast, 1,
            '(([rast1.val] - [rast2.val]) / ([rast1.val] + [rast2.val] + 0.000001))::float',
            '32BF'
        ) AS rast
        FROM verify_rasters r8, verify_rasters r4
        WHERE r8.name = 'b08_24' AND r4.name = 'b04_24'
    )
    SELECT ST_AsGDALRaster(
        ST_MapAlgebra(
            t2024.rast, 1, t2023.rast, 1,
            '([rast1.val] - [rast2.val])::float',
            '32BF'
        ),
        'GTiff'
    )
    FROM t2024, t2023;
    """
    print("[PostGIS] 正在执行 ST_MapAlgebra 两阶段代数运算...")
    cur.execute(mapalgebra_sql)
    pg_tif_bytes = cur.fetchone()[0]

    # 将 PostGIS 输出转回 NumPy
    import io
    with rasterio.open(io.BytesIO(pg_tif_bytes)) as pg_src:
        dndvi_pg = pg_src.read(1)

    # 6. 计算差值矩阵与统计指标
    abs_err = np.abs(dndvi_py - dndvi_pg)
    max_err = np.max(abs_err)
    rmse = np.sqrt(np.mean(abs_err ** 2))
    p50_err = np.percentile(abs_err, 50)
    p99_err = np.percentile(abs_err, 99)

    print("\n" + "=" * 60)
    print(">>> 验证结果汇报 (Correctness Report)")
    print("=" * 60)
    print(f"参与对比像元总数:  {abs_err.size:,}")
    print(f"最大绝对误差 (Max):  {max_err:.2e}")
    print(f"均方根误差 (RMSE):   {rmse:.2e}")
    print(f"中位数误差 (p50):    {p50_err:.2e}")
    print(f"99分位数误差 (p99):  {p99_err:.2e}")

    # 阈值判定: 单精度浮点运算容差标准 1e-4
    if max_err < 1e-4:
        print("\n [PASS] 验证通过！Python 与 PostGIS 差值计算在数学上完全等价。")
    else:
        print("\n [FAIL] 误差超出单精度允许范围，需检查代数表达式！")

    # 清理临时文件
    cur.execute("DROP TABLE verify_rasters;")
    cur.close()
    conn.close()
    for p in patch_paths.values():
        if os.path.exists(p): os.remove(p)

if __name__ == "__main__":
    run_verification()
