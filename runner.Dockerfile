FROM python:3.11-slim

# 安装编译与 GDAL 底层 C 库
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libspatialindex-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置 GDAL 环境变量
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# 安装 Python 科学计算与空间库栈
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    rasterio \
    geopandas \
    duckdb \
    shapely \
    pystac-client \
    planetary-computer \
    dask \
    psycopg2-binary \
    SQLAlchemy \
    geoalchemy2 \
    matplotlib \
    psutil

WORKDIR /workspace
