import os
import h5py
import numpy as np
from netCDF4 import Dataset

def nc4_to_mat(input_path, output_path):
    # 自动创建输出路径
    os.makedirs(output_path, exist_ok=True)

    file_list = [f for f in os.listdir(input_path) if f.endswith(".nc4")]

    # 生成固定网格
    latgrid = np.arange(89.95, -90, -0.1)
    longrid = np.arange(-179.95, 180, 0.1)
    Lon, Lat = np.meshgrid(longrid, latgrid)

    for filename in file_list:
        nc_path = os.path.join(input_path, filename)
        print(f"正在处理文件：{filename}")

        with Dataset(nc_path, mode='r') as ds:
            ku_vod_H = ds['vod_18h'][:]
            ku_vod_V = ds['vod_18v'][:]
            x_vod_H = ds['vod_10h'][:]
            x_vod_V = ds['vod_10v'][:]
            c_vod_H = ds['vod_06h'][:]
            c_vod_V = ds['vod_06v'][:]
            QC = ds['QC'][:]
            SM = ds['sm'][:]

        # 保存为 v7.3 mat 文件
        mat_path = os.path.join(output_path, filename + ".mat")
        with h5py.File(mat_path, 'w') as f:
            f.create_dataset("ku_vod_H", data=ku_vod_H)
            f.create_dataset("ku_vod_V", data=ku_vod_V)
            f.create_dataset("x_vod_H", data=x_vod_H)
            f.create_dataset("x_vod_V", data=x_vod_V)
            f.create_dataset("c_vod_H", data=c_vod_H)
            f.create_dataset("c_vod_V", data=c_vod_V)
            f.create_dataset("QC", data=QC)
            f.create_dataset("SM", data=SM)
            f.create_dataset("Lat", data=Lat)
            f.create_dataset("Lon", data=Lon)

        print("已保存为 v7.3 mat：", filename + ".mat")

    print("全部 nc4 文件已成功转换！")
