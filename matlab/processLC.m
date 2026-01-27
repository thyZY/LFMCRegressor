%% MCD12C1 CMG Landcover (0.1°) : .mat -> .h5 (keep variable names)
clc; clear;

inDir  = "G:\data\MCD12C1 CMG\01Degree\mat";
outDir = "G:\data\MCD12C1 CMG\01Degree\h5";

if ~exist(outDir, "dir")
    mkdir(outDir);
end

% 需要保存的变量名（保持原名不变）
vars = {'ENF','EBF','DNF','DBF','MF','CSH','OSH','WSA','SAV','GRA','WET','CRO','URB','CVM','SNO','BAR','WAT'};

files = dir(fullfile(inDir, "????001.mat"));

for i = 1:numel(files)

    inFile = fullfile(files(i).folder, files(i).name);

    % 从文件名提取 YYYY（YYYY001.mat）
    token = regexp(files(i).name, '^(\d{4})001\.mat$', 'tokens', 'once');
    if isempty(token)
        fprintf("[Skip] Bad name: %s\n", files(i).name);
        continue
    end
    YYYY = token{1};

    outFile = fullfile(outDir, [YYYY '001.h5']);
    fprintf("Processing %s -> %s\n", files(i).name, outFile);

    % 读取 mat
    S = load(inFile);

    % 若输出已存在，删除
    if exist(outFile, "file")
        delete(outFile);
    end

    % ===== 关键修复：先写入至少一个 dataset，保证 h5 文件被创建 =====
    firstVar = '';   % 用 char，避免 string/char 混用
    for v = 1:numel(vars)
        vn = vars{v}; % char
        if isfield(S, vn) && (isnumeric(S.(vn)) || islogical(S.(vn)))
            firstVar = vn;
            break
        end
    end

    if isempty(firstVar)
        fprintf("  [Skip] No valid variables found in %s\n", files(i).name);
        continue
    end

    % 写入第一个变量（创建 h5 文件）
    data0 = S.(firstVar);
    dset0 = ['/' firstVar];

    [dtype0, chunk0] = infer_h5_dtype_and_chunk(data0);

    h5create(outFile, dset0, size(data0), ...
        "Datatype", dtype0, ...
        "ChunkSize", chunk0, ...
        "Deflate", 4);
    h5write(outFile, dset0, data0);

    % ===== 现在文件已存在，可以写根属性 =====
    h5writeatt(outFile, "/", "Product", "MCD12C1 CMG Landcover fractions (IGBP)");
    h5writeatt(outFile, "/", "Resolution", "0.1 degree");
    h5writeatt(outFile, "/", "Temporal", "Annual");
    h5writeatt(outFile, "/", "Year", YYYY);
    h5writeatt(outFile, "/", "SourceMat", files(i).name);

    % 给第一个 dataset 写属性（可选）
    h5writeatt(outFile, dset0, "IGBP_Class_Fraction", firstVar);
    h5writeatt(outFile, dset0, "Units", "fraction");

    % ===== 写入其余变量 =====
    for v = 1:numel(vars)
        vn = vars{v}; % char

        % 用 strcmp 比较字符串，避免 == 的维度/类型问题
        if strcmp(vn, firstVar)
            continue
        end

        if ~isfield(S, vn)
            fprintf("  [Warn] Variable missing: %s\n", vn);
            continue
        end

        data = S.(vn);

        if ~(isnumeric(data) || islogical(data))
            fprintf("  [Warn] Variable not numeric/logical, skip: %s\n", vn);
            continue
        end

        dset = ['/' vn];

        [dtype, chunk] = infer_h5_dtype_and_chunk(data);

        h5create(outFile, dset, size(data), ...
            "Datatype", dtype, ...
            "ChunkSize", chunk, ...
            "Deflate", 4);

        h5write(outFile, dset, data);

        % dataset 属性（可选）
        h5writeatt(outFile, dset, "IGBP_Class_Fraction", vn);
        h5writeatt(outFile, dset, "Units", "fraction");
    end
end

fprintf("Done.\n");

%% ===== helper: infer dtype and chunk =====
function [dtype, chunk] = infer_h5_dtype_and_chunk(data)
    dtype = class(data); % 保持原始类型

    sz = size(data);
    if numel(sz) == 2
        chunk = [min(200, sz(1)), min(200, sz(2))];
    else
        chunk = ones(1, numel(sz));
        for k = 1:numel(sz)
            chunk(k) = min(50, sz(k));
        end
    end
end
