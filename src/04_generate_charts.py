import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = "/workspace/results"
PLOTS_DIR = "/workspace/results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# 统一学术风调色盘
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# ============================================================
# 图 1: 矢量算子三方压测对比 (Exp B: 100,000 规模)
# ============================================================
def plot_exp_b():
    csv_path = os.path.join(RESULTS_DIR, "exp_b_vector_queries.csv")
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    df_100k = df[df["n_parcels"] == 100000]

    engines = ["GeoPandas_STRtree", "DuckDB_Spatial", "PostGIS_GiST"]
    labels = ["GeoPandas (STRtree)", "DuckDB Spatial", "PostGIS (GiST)"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    sub = df_100k[df_100k["engine"].isin(engines)].set_index("engine").loc[engines]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(3)
    width = 0.25

    rects1 = ax.bar(x - width, sub["q1_intersects_s"], width, label=labels[0], color=colors[0], edgecolor="none")
    rects2 = ax.bar(x, sub["q2_dwithin_s"], width, label=labels[1], color=colors[1], edgecolor="none")
    rects3 = ax.bar(x + width, sub["q3_knn_s"], width, label=labels[2], color=colors[2], edgecolor="none")

    ax.set_yscale('log')
    ax.set_ylabel('Execution Time (Seconds, Log Scale)', fontsize=11, fontweight='bold')
    ax.set_title('Spatial Query Engine Benchmark (100,000 Parcels, 8,148 Rails)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(['Q1: Spatial Intersects', 'Q2: DWithin (500m)', 'Q3: 1-NN (Nearest Rail)'], fontsize=11)
    ax.legend(frameon=True, facecolor='#f8f9fa', edgecolor='none')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # 标注亮点
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.3f}s',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "fig1_vector_benchmark.png")
    plt.savefig(out)
    plt.close()
    print(f"[OK] 图 1 已生成: {out}")

# ============================================================
# 图 2: 端到端全链路流水线耗时阶段分布 (Exp D)
# ============================================================
def plot_exp_d():
    csv_path = os.path.join(RESULTS_DIR, "exp_d_end_to_end.csv")
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300, gridspec_kw={'width_ratios': [1.2, 1.8]})

    # 子图 1: 总耗时对比
    pipelines = ["Decoupled Hybrid", "All-in-DB"]
    totals = [
        df[df["pipeline"] == "Decoupled_Hybrid"]["Total_Wall_Time_s"].values[0],
        df[df["pipeline"] == "All_In_DB"]["Total_Wall_Time_s"].values[0]
    ]
    bars = ax1.bar(pipelines, totals, color=["#1b9e77", "#d95f02"], width=0.55)
    ax1.set_ylabel('Total Wall Time (Seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('End-to-End Pipeline Latency', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f'{h:.2f}s', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.annotate(f'15.98x Speedup\n(93.7% Time Saved)', xy=(0.5, 12), ha='center',
                 bbox=dict(boxstyle="round,pad=0.5", fc="#e6f5c9", ec="#1b9e77", lw=1.5), fontsize=10)

    # 子图 2: 阶段耗时堆叠分布 (百分比)
    stages_dec = [0.1104, 0.7693, 0.1209, 0.3788] # Compute, Vectorize, Ingest, Query
    stages_indb = [7.6304, 0.1987, 13.8776, 0.3281] # MapAlgebra, DumpPoly, RawIngest, Query

    labels = ['Compute / MapAlgebra', 'Vectorization', 'Ingest & Index', 'Spatial Analysis']
    palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    dec_pct = [s / sum(stages_dec) * 100 for s in stages_dec]
    indb_pct = [s / sum(stages_indb) * 100 for s in stages_indb]

    y = np.arange(2)
    left_dec = 0
    left_indb = 0
    for i in range(4):
        ax2.barh(0, dec_pct[i], left=left_dec, color=palette[i], label=labels[i] if i < 4 else "", height=0.45)
        ax2.barh(1, indb_pct[i], left=left_indb, color=palette[i], height=0.45)
        left_dec += dec_pct[i]
        left_indb += indb_pct[i]

    ax2.set_yticks(y)
    ax2.set_yticklabels(["Decoupled", "All-in-DB"], fontsize=11)
    ax2.set_xlabel('Stage Breakdown Percentage (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Time Allocation Breakdown', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=True)
    ax2.set_xlim(0, 100)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "fig2_end_to_end_breakdown.png")
    plt.savefig(out)
    plt.close()
    print(f"[OK] 图 2 已生成: {out}")

if __name__ == "__main__":
    plot_exp_b()
    plot_exp_d()
    print(">>> 全部成果图表渲染完毕！")
