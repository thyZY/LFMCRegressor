%% ESA CCI AGB: GlobBiomassYYYY.mat -> GlobBiomassYYYY.h5 (variable: AGB)
clc; clear;

inDir  = "G:\data\ESACCI AGB\GlobBiomass_mat";
outDir = "G:\data\ESACCI AGB\h5";

if ~exist(outDir, "dir")
    mkdir(outDir);
end

files = dir(fullfile(inDir, "GlobBiomass????.mat"));

for i = 1:numel(files)

    inFile = fullfile(files(i).folder, files(i).name);

    % 解析年份（GlobBiomassYYYY.mat）
    tok = regexp(files(i).name, '^GlobBiomass(\d{4})\.mat$', 'tokens', 'once');
    if isempty(tok)
        fprintf("[Skip] Bad name: %s\n", files(i).name);
        continue
    end
    YYYY = tok{1};

    outFile = fullfile(outDir, "GlobBiomass" + string(YYYY) + ".h5");
    fprintf("Processing %s -> %s\n", files(i).name, outFile);

    % 读取 MAT
    S = load(inFile);

    if ~isfield(S, "AGB")
        fprintf("  [Skip] Variable 'AGB' not found in %s\n", files(i).name);
        continue
    end

    AGB = S.AGB;

    if ~(isnumeric(AGB) || islogical(AGB))
        fprintf("  [Skip] 'AGB' is not numeric/logical in %s\n", files(i).name);
        continue
    end

    % 若输出已存在，删除
    if exist(outFile, "file")
        delete(outFile);
    end

    % ===== 先写 dataset（创建 h5 文件）=====
    dset = "/AGB";

    [dtype, chunk] = infer_h5_dtype_and_chunk(AGB);

    h5create(outFile, dset, size(AGB), ...
        "Datatype", dtype, ...
        "ChunkSize", chunk, ...
        "Deflate", 4);

    h5write(outFile, dset, AGB);

    % ===== 写根属性（可选）=====
    h5writeatt(outFile, "/", "Product", "ESA CCI AGB (GlobBiomass)");
    h5writeatt(outFile, "/", "Variable", "AGB");
    h5writeatt(outFile, "/", "Year", YYYY);
    h5writeatt(outFile, "/", "SourceMat", files(i).name);

    % dataset 属性（可选）
    h5writeatt(outFile, dset, "Name", "AGB");

end

fprintf("Done.\n");

%% ===== helper: infer dtype and chunk =====
function [dtype, chunk] = infer_h5_dtype_and_chunk(data)
    dtype = class(data); % 保持原始类型

    sz = size(data);

    % 保证 ChunkSize 的维度数与数据一致，且每个维度 <= 对应数据维度
    if numel(sz) == 2
        chunk = [min(200, sz(1)), min(200, sz(2))];
    else
        chunk = ones(1, numel(sz));
        for k = 1:numel(sz)
            chunk(k) = min(50, sz(k));
        end
    end
end
