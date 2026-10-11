# Spatial Computing Architecture Benchmark: Decoupled Streaming Pipeline vs. In-Database Batch Processing
**Benchmarking on Tokyo Sentinel-2 Imagery and 100k Spatial Parcel Joins**

## 1. Abstract & Scope (Research Boundaries & Data Disclosure)
Addressing the long-standing architectural debate between **"Compute-in-DB" (pushing band algebra into RDBMS)** and **"Decoupled Streaming Pipeline" (external compute + pushdown storage)** in the remote sensing and GIS domains, this project implements an end-to-end benchmark system constrained to **4 vCPUs and 3.8 GB RAM**.

To ensure reproducibility and objective fidelity, the datasets are delineated as follows:
1. **Multi-temporal Raster Imagery (Real-world Data)**: Sentinel-2 L2A tiles provided by Microsoft Planetary Computer (MGRS Tile: `54SUE`, acquisitions on 2023-07-26 and 2024-08-04, cloud-free aligned subset of $3,299 \times 3,258$ pixels, projected to UTM Zone 54N).
2. **Railway Trunk Network (Real-world Data)**: 8,148 real railway segments covering central Tokyo, extracted via OpenStreetMap.
3. **Spatial Parcel Baseline (Synthetic Benchmark Data)**: 100,000 synthetic building/cropland polygons generated within Tokyo's geographical bounding box (deterministic seed `seed=42`, edge lengths 15–40 meters).
4. **Change Detection Patches (Explicitly Partitioned)**:
   - **Experiments B & C**: 5,000 fixed 50-meter synthetic patches to keep spatial geometry complexity strictly controlled as a constant variable.
   - **Experiment D**: **9,830 real vegetation loss patches** extracted directly from real Sentinel-2 imagery using a threshold of $\Delta\text{NDVI} < -0.15$.

---

## 2. Benchmark Experiments & Analytical Findings

### 2.1 Experiment 0: Pixel-level Operator Mathematical Parity Check
Verified NumPy vectorization against PostGIS `ST_MapAlgebra` on a $512 \times 512$ tile:
- **Evaluated Pixels**: 262,144
- **Max Absolute Error**: $< 1.0 \times 10^{-6}$
- **Conclusion**: Confirms that downstream performance deltas stem entirely from system runtime and storage engine overheads, ruling out mathematical or algorithmic discrepancies.

---

### 2.2 Experiment A: Raster Algebra Compute Speed & WAL Write Overhead (Exp A)
Evaluated red and near-infrared (NIR) band differencing across two acquisitions (NDVI change extraction). Measured median wall-clock runtime, WAL volume generated, and memory/disk footprints:

| Scale (Pixels) | Approach & Tile Configuration | Wall-Clock Time (s) | Speedup Ratio | WAL Generated (MB) | Table / Memory Footprint (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1x (260k px)** | Python (NumPy) | **0.0032s** | **Baseline (572x)** | **0.0 MB** | 1.25 MB (RAM) |
| | PostGIS In-DB (32×32) | 1.9674s | 1.0x | 1.26 MB | 1.38 MB (Disk) |
| | PostGIS In-DB (128×128)| 1.8517s | 1.06x | 1.20 MB | 1.07 MB (Disk) |
| | PostGIS In-DB (256×256)| 1.8310s | 1.07x | 1.27 MB | 1.07 MB (Disk) |
| **4x (1.05M px)** | Python (NumPy) | **0.0145s** | **Baseline (488x)** | **0.0 MB** | 5.00 MB (RAM) |
| | PostGIS In-DB (32×32) | 7.5590s | 1.0x | 4.66 MB | 5.48 MB (Disk) |
| | PostGIS In-DB (128×128)| 7.1462s | 1.06x | 4.42 MB | 4.20 MB (Disk) |
| | PostGIS In-DB (256×256)| 7.0730s | 1.07x | 4.40 MB | 4.20 MB (Disk) |
| **16x (4.19M px)**| Python (NumPy) | **0.0470s** | **Baseline (605x)** | **0.0 MB** | 20.00 MB (RAM) |
| | PostGIS In-DB (32×32) | 30.4489s | 1.0x | 18.32 MB | 21.87 MB (Disk) |
| | PostGIS In-DB (128×128)| 29.3655s | 1.04x | 17.32 MB | 16.72 MB (Disk) |
| | PostGIS In-DB (256×256)| 28.4649s | 1.07x | 17.32 MB | 16.71 MB (Disk) |

**Key Insights:**
1. **Raw Compute Disparity**: Vectorized NumPy computations leverage SIMD vector instructions and contiguous cache-friendly memory layouts, consistently outperforming PostGIS `ST_MapAlgebra` by **500x to 600x**.
2. **Write Amplification & WAL Overhead**: When PostGIS writes derived raster tiles to disk, Write-Ahead Log (WAL) generation scales strictly linearly. Processing 4.19M pixels generated **17.32 MB of WAL**, proving that ingesting high-frequency raster algebra results places severe stress on disk I/O and replication bandwidth.
3. **Tile Sizing Overhead**: $32\times32$ micro-tiles incur a ~7% runtime penalty compared to $256\times256$ tiles due to the excessive per-tuple header overhead in PostgreSQL.

---

### 2.3 Experiment B: Vector Spatial Predicate Stress Test (Exp B - 100k Parcels)
| Operator | GeoPandas (STRtree) | DuckDB Spatial | PostGIS (GiST) | PostGIS (Unindexed Baseline) |
| :--- | :--- | :--- | :--- | :--- |
| **Q1: Intersects** | **0.0147s** | 0.0440s | 0.2172s | 9.6892s (44x slower) |
| **Q2: Within 500m (DWithin)** | 0.5534s | **0.0707s** | 0.2503s | 39.5796s (158x slower) |
| **Q3: 1-NN Nearest Neighbor (KNN)** | 0.0532s | 1.9187s | **0.0458s** | 5.3691s (117x slower) |

**Key Insights:**
1. **In-Memory Query Engine Strengths**: GeoPandas and DuckDB benefit from in-process zero-copy memory reads, achieving single-digit millisecond latency on simple intersections. However, GeoPandas regresses sharply on distance filters (0.55s) due to the overhead of materializing polygon buffers (`.buffer()`).
2. **Production-Grade Value of Spatial Indexes**: PostGIS GiST index achieves top latency in 1-NN queries via its `<->` bounding-box distance priority queue traversal (**0.0458s**)—delivering a **117x speedup** over the unindexed baseline and outperforming DuckDB Spatial (which lacks dedicated KNN index support) by **42x**.

---

### 2.4 Experiment C: Concurrent Read Interference & MVCC Isolation (Exp C)
Under 4 concurrent background client workers executing complex vector queries (over 25 seconds, >6,000 transactions total), we measured read latency interference:

| Scenario | Throughput (QPS) | p50 Latency | p95 Latency | p99 Latency | Max Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1_Baseline_Idle (No interference)** | **253.20** | **15.45 ms** | **17.58 ms** | **24.80 ms** | **45.28 ms** |
| **2_Coupled_InDB (In-DB raster batch)**| 251.92 | 15.49 ms | 17.88 ms | 24.53 ms | 52.52 ms |
| **3_Decoupled (External CPU-bound Python)**| 242.08 | 15.91 ms | 18.80 ms | 25.69 ms | 72.68 ms |

**Counter-Intuitive Discovery & Analysis:**
Under read-only concurrency, **PostgreSQL’s MVCC and kernel scheduler effectively isolate raster batch compute from vector read queries**. In-DB raster batch processing did not trigger p99 tail latency spikes (both p50 and p99 stayed within a 1% variance).

This demonstrates that **the primary bottleneck of In-Database raster compute is not transaction starvation of concurrent reads, but rather its abysmal single-node computational efficiency (the 600x gap in Exp A) and the massive physical I/O / WAL write amplification during persistence.**

---

### 2.5 Experiment D: End-to-End Pipeline Comparison (Exp D - 1024×1024 Real Extraction)
Extracting 9,830 real vegetation change patches and performing spatial collision analysis against 100,000 parcels:

* **Stage Breakdown**:
  1. **Compute & Vectorize Phase**:
     - Decoupled Pipeline: $0.1104\text{s} + 0.7693\text{s} = \mathbf{0.8797\text{s}}$
     - Pure In-DB Pipeline: $7.6304\text{s} + 0.1987\text{s} = \mathbf{7.8291\text{s}}$
     - **Compute Speedup: 8.90x**
  2. **Ingestion & Indexing Phase**:
     - Decoupled pipeline uses streaming `COPY` with Hex-WKB bypass, achieving **81,339 polygons/second** (0.1209s total).
     - In-DB pipeline incurs 0.5–1.0s depending on row serialization style.
  3. **Spatial Join Phase**:
     - Decoupled pipeline: 0.3788s; In-DB pipeline: 0.3281s. Both complete spatial joins against 100k parcels in ~0.3 seconds.
* **Conclusion**: Even with fully optimized ingestion paths, the decoupled architecture sustains an overall **~6x to ~8x end-to-end throughput advantage**.

---

## 3. Architectural Takeaways

1. **Say No to Heavy Raster Compute In-Database**: Avoid storing high-resolution satellite imagery or high-frequency raster rasters in relational DB tables for map algebra. NumPy / Xarray provide a >500x compute advantage while entirely avoiding megabyte-scale WAL write amplification.
2. **Leverage PostGIS as the Spatial Relational Hub**: PostGIS excels at persisting complex vector entities, guaranteeing ACID compliance, and executing lightning-fast topological joins and GiST-indexed spatial queries (especially k-NN).
3. **The Best Practice Pattern**: **"Compute outside via vectorized streams, push via streaming COPY pipelines, index and query inside PostGIS."**
