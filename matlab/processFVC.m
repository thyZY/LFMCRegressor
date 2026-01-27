%% GLASS FVC (0.05° -> 0.1° by 2x2 valid-mean)  -> save as .h5
clc; clear;

inRoot  = "G:\data\GLASS FVC\GLASS FVC\0.05D";
outRoot = "G:\data\GLASS FVC\01nc";   % 你说“其余不变”，这里保持原变量名/路径不动

if ~exist(outRoot, 'dir')
    mkdir(outRoot);
end

% ===== 自动获取所有 YYYY 文件夹（如果没有年份子文件夹，也会直接扫根目录）=====
dirs = dir(inRoot);
dirs = dirs([dirs.isdir]);

yearDirs = {};
for d = 1:numel(dirs)
    dirname = dirs(d).name;
    if strcmp(dirname, '.') || strcmp(dirname, '..')
        continue
    end
    if ~isempty(regexp(dirname, '^\d{4}$', 'once'))
        yearDirs{end+1} = fullfile(inRoot, dirname); %#ok<AGROW>
    end
end

% 如果没有年份文件夹，就直接处理 inRoot 下的文件
if isempty(yearDirs)
    yearDirs = {inRoot};
end

% ===== 主循环 =====
for yd = 1:numel(yearDirs)

    inDir = yearDirs{yd};
    fprintf("Scanning: %s\n", inDir);

    files = dir(fullfile(inDir, "GLASS10B01*.hdf"));

    for i = 1:numel(files)

        inFile = fullfile(files(i).folder, files(i).name);

        % ===== 从文件名中提取 YYYY 和 DOY，然后转 YYYYMMDD =====
        % 文件示例：GLASS10B01.V40.AYYYYDOY.2025154.hdf
        % 注意：A之后的字符串不固定 -> 只抓 A 后面的 YYYY(4) + DOY(3)
        token = regexp(files(i).name, 'A(\d{4})(\d{3})\.', 'tokens', 'once');
        if isempty(token)
            fprintf("  [Skip] Cannot parse date from: %s\n", files(i).name);
            continue
        end

        YYYY = str2double(token{1});
        DOY  = str2double(token{2});

        % DOY -> YYYYMMDD
        dt = datenum(YYYY, 1, 1) + DOY - 1;
        YYYYMMDD = string(datestr(dt, 'yyyymmdd'));

        % ===== 输出改为 .h5 =====
        outFile = fullfile(outRoot, YYYYMMDD + ".h5");
        fprintf("  -> %s  (%s)\n", YYYYMMDD, files(i).name);

        % ===== 读取 SDS: FVC =====
        % 直接按 SDS 名称读取（若遇到特殊结构，可再改为遍历 hdfinfo 的方式）
        try
            fvc_raw = hdfread(inFile, "FVC");
        catch ME
            fprintf("  [Skip] Failed to read SDS 'FVC' from %s\n    %s\n", files(i).name, ME.message);
            continue
        end

        fvc_raw = double(fvc_raw);

        % ===== 原始有效值范围 0-250（未缩放）=====
        fvc_raw(fvc_raw < 0 | fvc_raw > 250) = NaN;

        % ===== 缩放因子 0.004 =====
        fvc_005 = fvc_raw * 0.004;   % 0.05°，尺寸应为 3600×7200

        % ===== 0.05° -> 0.1°：2×2 网格仅对有效值求平均 =====
        % 规则：2×2 中只对有效值取平均；若全无效 -> NaN
        if size(fvc_005,1) ~= 3600 || size(fvc_005,2) ~= 7200
            fprintf("  [Warn] Unexpected size %dx%d (expect 3600x7200). Still try 2x2...\n", ...
                size(fvc_005,1), size(fvc_005,2));
        end

        fvc_01 = blockMean2x2_valid(fvc_005); % 输出 1800×3600

        % ===== 写 HDF5（YYYYMMDD.h5）=====
        if exist(outFile, 'file')
            delete(outFile)
        end

        dset = "/FVC";

        % 可选：压缩与分块（更省空间/更适合 Python 分块读）
        chunkSize = [200 200];
        h5create(outFile, dset, size(fvc_01), ...
            "Datatype", "double", ...
            "ChunkSize", chunkSize, ...
            "Deflate", 4);

        h5write(outFile, dset, fvc_01);

        % ===== 属性（写到文件根组 /，与原 ncwriteatt 类似）=====
        h5writeatt(outFile, "/", "Product", "GLASS FVC");
        h5writeatt(outFile, "/", "Resolution", "0.1 degree (aggregated from 0.05 degree by 2x2 valid-mean)");
        h5writeatt(outFile, "/", "SDS_Name", "FVC");
        h5writeatt(outFile, "/", "ScaleFactor", 0.004);
        h5writeatt(outFile, "/", "ValidRangeRaw", "0-250 (before scaling)");
        h5writeatt(outFile, "/", "Date", char(YYYYMMDD));
        h5writeatt(outFile, "/", "SourceFile", files(i).name);

        % 也可以把行列维度写成属性（方便 Python 判断）
        h5writeatt(outFile, dset, "Rows", int32(size(fvc_01,1)));
        h5writeatt(outFile, dset, "Cols", int32(size(fvc_01,2)));

    end
end

%% ====== local function: 2x2 valid mean ======
function out = blockMean2x2_valid(A)
    r = size(A,1);
    c = size(A,2);

    r2 = floor(r/2)*2;
    c2 = floor(c/2)*2;
    A = A(1:r2, 1:c2);

    a11 = A(1:2:end, 1:2:end);
    a21 = A(2:2:end, 1:2:end);
    a12 = A(1:2:end, 2:2:end);
    a22 = A(2:2:end, 2:2:end);

    v11 = ~isnan(a11); v21 = ~isnan(a21);
    v12 = ~isnan(a12); v22 = ~isnan(a22);

    sumv = zeros(size(a11));
    cnt  = zeros(size(a11));

    sumv(v11) = sumv(v11) + a11(v11); cnt(v11) = cnt(v11) + 1;
    sumv(v21) = sumv(v21) + a21(v21); cnt(v21) = cnt(v21) + 1;
    sumv(v12) = sumv(v12) + a12(v12); cnt(v12) = cnt(v12) + 1;
    sumv(v22) = sumv(v22) + a22(v22); cnt(v22) = cnt(v22) + 1;

    out = sumv ./ cnt;
    out(cnt == 0) = NaN;
end
