# extract_lai.py
import os
import numpy as np
from netCDF4 import Dataset
import h5py
from datetime import datetime, timedelta

def hdf_to_lai_mat(input_folder, output_folder):
    """
    批量处理 GLASS LAI HDF 文件，提取 LAI 数据，2*2 下采样，
    生成 Lat、Lon 网格，保存为 MATLAB v7.3 格式 mat 文件。

    参数：
        input_folder: str, HDF 文件夹路径
        output_folder: str, MAT 文件保存路径
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    hdf_files = [f for f in os.listdir(input_folder) if f.endswith(".hdf")]

    for hdf_file in hdf_files:
        hdf_path = os.path.join(input_folder, hdf_file)
        print("处理文件:", hdf_path)

        # 打开 HDF 文件
        ds = Dataset(hdf_path, 'r')
        lai_raw = ds.variables['LAI'][:].astype(float)
        fill_value = ds.variables['LAI']._FillValue
        scale_factor = ds.variables['LAI'].scale_factor

        # 替换填充值为 NaN
        lai_raw[lai_raw == fill_value] = np.nan

        # 应用缩放因子
        lai_scaled = lai_raw * scale_factor

        # 2x2 平均下采样（向量化实现）
        h, w = lai_scaled.shape
        h2, w2 = h // 2, w // 2
        lai_down = np.empty((h2, w2), dtype=float)

        # 分块求平均，去掉 NaN
        block = lai_scaled[:h2*2, :w2*2].reshape(h2, 2, w2, 2)
        # 计算每个 2x2 块平均值
        with np.errstate(invalid='ignore'):
            lai_down = np.nanmean(block, axis=(1, 3))

        # 如果整块都是 NaN，nanmean 会返回 NaN，符合要求

        # 生成 Lat 和 Lon
        Lat = np.linspace(89.95, -89.95, h2).reshape(h2, 1)
        Lat = np.repeat(Lat, w2, axis=1)
        Lon = np.linspace(-179.95, 179.95, w2).reshape(1, w2)
        Lon = np.repeat(Lon, h2, axis=0)

        # 从 HDF 文件名提取日期 YYYYMMDD
        # 文件名示例: GLASS01B01.V60.AYYYY{DOY}.2024107.hdf
        try:
            base_name = os.path.basename(hdf_file)
            parts = base_name.split('.')
            yyyydoy = parts[2][1:]  # 去掉前面的 'A'
            year = int(yyyydoy[:4])
            doy = int(yyyydoy[4:])
            date = datetime(year, 1, 1) + timedelta(days=doy-1)
            date_str = date.strftime("%Y-%m-%d")
            mat_name = f"{date_str}.tif.mat"
        except Exception as e:
            print("日期解析失败:", e)
            mat_name = base_name + ".mat"

        mat_path = os.path.join(output_folder, mat_name)

        # 保存为 MATLAB v7.3 格式（HDF5）
        with h5py.File(mat_path, 'w') as f:
            f.create_dataset('lai', data=lai_down, compression='gzip')
            f.create_dataset('Lat', data=Lat, compression='gzip')
            f.create_dataset('Lon', data=Lon, compression='gzip')

        print("已保存:", mat_path)

