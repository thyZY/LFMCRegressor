%% CanopyHeight: CH.mat -> CH.h5 (variable: Hveg)
clc; clear;

inFile  = "G:\data\CanopyHeight\CH.mat";
outFile = "G:\data\CanopyHeight\CH.h5";

% 读取 MAT
S = load(inFile);

if ~isfield(S, "Hveg")
    error("Input MAT does not contain variable 'Hveg'.");
end

Hveg = S.Hveg;

if ~(isnumeric(Hveg) || islogical(Hveg))
    error("'Hveg' is not numeric/logical, cannot write to HDF5.");
end

% 若输出已存在，删除
if exist(outFile, "file")
    delete(outFile);
end

% ===== 先写 dataset（创建 h5 文件）=====
dset = "/Hveg";

[dtype, chunk] = infer_h5_dtype_and_chunk(Hveg);

h5create(outFile, dset, size(Hveg), ...
    "Datatype", dtype, ...
    "ChunkSize", chunk, ...
    "Deflate", 4);

h5write(outFile, dset, Hveg);

% ===== 写根属性（可选）=====
h5writeatt(outFile, "/", "Product", "Canopy Height");
h5writeatt(outFile, "/", "Variable", "Hveg");
h5writeatt(outFile, "/", "SourceMat", "CH.mat");

% dataset 属性（可选）
h5writeatt(outFile, dset, "Name", "Hveg");

fprintf("Done: %s\n", outFile);

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
