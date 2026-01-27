import os
import subprocess

def unzip_rar_by_year(winrar_exe, rar_path, unzip_path, start_year, end_year):
    """
    根据 rar_path 文件夹名称判断前缀（Asc 或 Des），解压指定年份范围内的 RAR 文件。
    """
    folder_name = os.path.basename(os.path.normpath(rar_path))

    if 'Asc' in folder_name:
        prefix = 'Asc'
    elif 'Des' in folder_name:
        prefix = 'Des'
    else:
        raise ValueError(f"无法识别文件夹前缀: {folder_name}")

    for year in range(start_year, end_year + 1):
        rar_file = os.path.join(rar_path, f"{prefix}_{year}.rar")
        out_dir = os.path.join(unzip_path, str(year))

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if not os.path.isfile(rar_file):
            print(f"文件不存在: {rar_file}, 跳过...")
            continue

        command = f'"{winrar_exe}" e -y "{rar_file}" "{out_dir}"'
        print("Executing:", command)
        status = subprocess.call(command, shell=True)
        print("解压成功" if status == 0 else "解压失败:", rar_file)