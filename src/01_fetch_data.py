import os
import geopandas as gpd
import planetary_computer as pc
from pystac_client import Client
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

TOKYO_BBOX = [139.56, 35.53, 139.92, 35.82]  # WGS84
DATA_DIR = "/data/raw_rasters"
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_aligned_sentinel():
    print(">>> 正在连接 Microsoft Planetary Computer STAC...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    periods = {
        "2023": "2023-07-01/2023-08-31",
        "2024": "2024-07-01/2024-08-31",
    }

    reference_window = None
    reference_transform = None
    reference_shape = None

    for year, dtrange in periods.items():
        print(f"\n正在检索 {year} 年夏季 Sentinel-2 L2A 影像 (锁定 54SUE 瓦片)...")
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=TOKYO_BBOX,
            datetime=dtrange,
            query={
                "s2:mgrs_tile": {"eq": "54SUE"},  # 强行锁定东京都核心瓦片
                "eo:cloud_cover": {"lt": 15},
            },
            sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
            limit=1,
        )
        items = list(search.items())
        if not items:
            raise RuntimeError(f"{year} 年在 54SUE 瓦片下未找到少云影像！")

        item = items[0]
        print(
            f"选定景 ID: {item.id}, 拍摄日期: {item.datetime.date()}, 云量: {item.properties['eo:cloud_cover']:.2f}%"
        )

        for band in ["B04", "B08"]:
            out_path = os.path.join(DATA_DIR, f"{year}_{band}_tokyo.tif")
            if os.path.exists(out_path):
                print(f"文件已存在，跳过: {out_path}")
                # 如果是 2023 读取作为基准
                if reference_shape is None:
                    with rasterio.open(out_path) as ref_src:
                        reference_shape = (ref_src.height, ref_src.width)
                        reference_transform = ref_src.transform
                continue

            asset_url = item.assets[band].href
            print(f"正在拉取 {year} {band}...")
            with rasterio.open(asset_url) as src:
                if reference_window is None:
                    src_bounds = transform_bounds(
                        "EPSG:4326", src.crs, *TOKYO_BBOX
                    )
                    reference_window = from_bounds(
                        *src_bounds, transform=src.transform
                    )
                    # 取整像元窗口，避免浮点裁剪出现 1 像素偏差
                    reference_window = reference_window.round_offsets().round_lengths()

                data = src.read(1, window=reference_window)
                win_transform = src.window_transform(reference_window)

                if reference_shape is None:
                    reference_shape = data.shape
                    reference_transform = win_transform
                else:
                    # 严防像元对不齐
                    assert (
                        data.shape == reference_shape
                    ), f"尺寸失配: {data.shape} vs {reference_shape}"

                profile = src.profile.copy()
                profile.update(
                    {
                        "height": data.shape[0],
                        "width": data.shape[1],
                        "transform": win_transform,
                        "compress": "deflate",
                    }
                )

                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(data, 1)

            print(
                f"已保存: {out_path} (精确对齐大小: {data.shape[1]}x{data.shape[0]} 像元)"
            )


if __name__ == "__main__":
    fetch_aligned_sentinel()
