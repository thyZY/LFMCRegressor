# extract_LC.py
import os
import numpy as np
from netCDF4 import Dataset
import scipy.io as sio

# 地物类型简写（顺序对应 Land_Cover_Type_1_Percent 第一维）
LC_CLASSES = [
    "WAT", "ENF", "EBF", "DNF", "DBF", "MF", "CSH", "OSH",
    "WSA", "SAV", "GRA", "WET", "CRO", "URB", "CVM", "SNO", "BAR"
]

def process_lc_folder(input_folder, output_folder):
    """
    批量处理 Land Cover HDF 文件，将 Land_Cover_Type_1_Percent 17层数据：
      - 替换 255 为 NaN
      - 除以100转为 0-1
      - 2x2 下采样平均（NaN 自动忽略）
    每个类别单独保存为 MATLAB 变量。
    
    参数：
        input_folder: HDF 文件夹路径
        output_folder: MAT 文件保存路径
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    hdf_files = [f for f in os.listdir(input_folder) if f.endswith(".hdf")]
    
    for hdf_file in hdf_files:
        hdf_path = os.path.join(input_folder, hdf_file)
        print("处理文件:", hdf_path)

        # 打开 HDF 文件
        ds = Dataset(hdf_path, 'r')
        lc_raw = ds.variables['Land_Cover_Type_1_Percent'][:]  # shape: (3600, 7200, 17)
        ds.close()

        # 转换为 float32 避免整数运算问题
        lc_raw = lc_raw.astype(np.float32)

        # 替换填充值 255 为 NaN
        lc_raw[lc_raw == 255] = np.nan

        # 转换为 0-1
        lc_raw /= 100.0

        # 转置维度为 (17, 3600, 7200)
        lc_raw = np.transpose(lc_raw, (2,0,1))

        # 2x2 下采样向量化
        n_layers, h, w = lc_raw.shape
        h2, w2 = h//2, w//2
        lc_down = np.nanmean(
            lc_raw[:, :h2*2, :w2*2].reshape(n_layers, h2, 2, w2, 2),
            axis=(2,4)
        )  # shape: (17, 1800, 3600)

        # 将每个类别单独存入字典
        mat_dict = {}
        for i, cls in enumerate(LC_CLASSES):
            print(f"  正在处理类别 {i} ({cls}) ...")
            mat_dict[cls] = lc_down[i]

        # 保存为 MATLAB 可直接 load 的 MAT 文件
        try:
            base_name = os.path.basename(hdf_file)
            year = base_name.split('.')[1][1:]  # 'AYYYY' -> 'YYYY'
            mat_name = f"{year}.mat"
        except Exception as e:
            print("年份解析失败:", e)
            mat_name = base_name + ".mat"

        mat_path = os.path.join(output_folder, mat_name)
        sio.savemat(mat_path, mat_dict, do_compression=True)

        print("已保存:", mat_path)
        print("-"*50)
