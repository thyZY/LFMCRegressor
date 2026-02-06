function processHveg()
% processHveg
% ------------------------------------------------------------
% 拼接多个 4-band GeoTIFF 分幅（可能存在 301x300/301x301、边界非整数、
% 小范围重叠等情况），生成固定 0.1° 全球网格 (1800x3600) 的 HDF5：
%
% 目标网格中心：
%   lat centers:  89.95, 89.85, ..., -89.95   (1800 rows)
%   lon centers: -179.95, -179.85, ..., 179.95 (3600 cols)
%
% 映射规则（最近邻）：
%   row = round((89.95 - lat)/0.1) + 1
%   col = round((lon - (-179.95))/0.1) + 1
%
% 重叠处理策略：
%   只填充目标数组中仍为 NaN 的位置（避免重叠处 NoData 覆盖有效值）
%
% 输出：
%   G:\data\CanopyHeight\ETH_global_reprocess\Hveg_01Deg.h5
%   datasets:
%     /H_forest, /H_shrub, /H_herb, /H_mix   (single, 1800x3600)
%     /lat_center (1800x1), /lon_center (3600x1)
% ------------------------------------------------------------

%% ---------------- 0) 路径配置 ----------------
tileDir = "G:\data\CanopyHeight\ETH_global_reprocess\Tiles";
outH5   = "G:\data\CanopyHeight\ETH_global_reprocess\Hveg_01Deg.h5";

tifs = dir(fullfile(tileDir, "*.tif"));
if isempty(tifs)
    error("未在目录中找到 tif：%s", tileDir);
end
fprintf("Found %d tif tiles in: %s\n", numel(tifs), tileDir);

%% ---------------- 1) 构造目标网格 ----------------
latC = single(89.95 : -0.1 : -89.95);        % 1x1800
lonC = single(-179.95 : 0.1 : 179.95);       % 1x3600

nLat = numel(latC);  % 1800
nLon = numel(lonC);  % 3600

% 4 个空数组
H_forest = single(nan(nLat, nLon));
H_shrub  = single(nan(nLat, nLon));
H_herb   = single(nan(nLat, nLon));
H_mix    = single(nan(nLat, nLon));

%% ---------------- 2) 循环读取分幅并灌格 ----------------
for k = 1:numel(tifs)
    fp = fullfile(tifs(k).folder, tifs(k).name);
    fprintf("[%d/%d] %s\n", k, numel(tifs), tifs(k).name);

    % 读多波段tif + 地理参考
    try
        [A, R] = read_geotiff_multiband(fp);
    catch ME
        warning("读取失败：%s (%s)", fp, ME.message);
        continue;
    end

    if ndims(A) ~= 3 || size(A,3) < 4
        warning("跳过：%s (band=%d，不是4波段)", fp, size(A,3));
        continue;
    end

    A = single(A);

    % 常见 nodata 处理（保守）
    A(A <= -3.0e30) = nan;
    % 如果你确认 nodata 是 -9999，可以取消注释：
    % A(A == -9999) = nan;

    nRows = size(A,1);
    nCols = size(A,2);

    % 计算该 tif 每个像元中心经纬度（rows x cols）
    [latPix, lonPix] = pixel_center_latlon(R, nRows, nCols);

    % lon wrap 到 [-180,180)
    lonPix = mod(lonPix + 180, 360) - 180;

    % 映射到目标网格索引 (nearest)
    rowIdx = round((89.95 - latPix) ./ 0.1) + 1;               % 1..1800
    colIdx = round((lonPix - (-179.95)) ./ 0.1) + 1;           % 1..3600

    % 边界保护
    rowIdx = min(max(rowIdx, 1), nLat);
    colIdx = min(max(colIdx, 1), nLon);

    % 线性索引
    lin = sub2ind([nLat, nLon], rowIdx(:), colIdx(:));

    % 逐波段写入（只填 NaN 的目标像元）
    H_forest = fill_nan_only(H_forest, lin, A(:,:,1));
    H_shrub  = fill_nan_only(H_shrub,  lin, A(:,:,2));
    H_herb   = fill_nan_only(H_herb,   lin, A(:,:,3));
    H_mix    = fill_nan_only(H_mix,    lin, A(:,:,4));
end

%% ---------------- 3) 写出 HDF5 ----------------
if exist(outH5, "file")
    delete(outH5);
end

chunk = [180, 360]; % 合理 chunk（可按 IO 调）

h5create(outH5, "/H_forest", [nLat nLon], "Datatype","single", "ChunkSize",chunk, "Deflate",4);
h5create(outH5, "/H_shrub",  [nLat nLon], "Datatype","single", "ChunkSize",chunk, "Deflate",4);
h5create(outH5, "/H_herb",   [nLat nLon], "Datatype","single", "ChunkSize",chunk, "Deflate",4);
h5create(outH5, "/H_mix",    [nLat nLon], "Datatype","single", "ChunkSize",chunk, "Deflate",4);

h5write(outH5, "/H_forest", H_forest);
h5write(outH5, "/H_shrub",  H_shrub);
h5write(outH5, "/H_herb",   H_herb);
h5write(outH5, "/H_mix",    H_mix);

% 保存中心点数组
h5create(outH5, "/lat_center", [nLat 1], "Datatype","single");
h5create(outH5, "/lon_center", [nLon 1], "Datatype","single");
h5write(outH5, "/lat_center", latC(:));
h5write(outH5, "/lon_center", lonC(:));

% 统计（不使用 mean，避免被遮蔽）
fprintf("\nSaved: %s\n", outH5);
fprintf("NaN ratio forest: %.4f\n", nan_ratio(H_forest));
fprintf("NaN ratio shrub : %.4f\n", nan_ratio(H_shrub));
fprintf("NaN ratio herb  : %.4f\n", nan_ratio(H_herb));
fprintf("NaN ratio mix   : %.4f\n", nan_ratio(H_mix));
fprintf("DONE.\n");

end

%% ====================== 子函数区 ======================

function r = nan_ratio(X)
% 返回 NaN 占比（不用 mean，避免函数遮蔽）
n = numel(X);
if n == 0
    r = NaN;
    return;
end
r = double(sum(isnan(X(:)))) / double(n);
end

function [A, R] = read_geotiff_multiband(fp)
% 兼容不同 MATLAB 版本读取多波段 GeoTIFF
try
    [A, R] = readgeoraster(fp);  % 新版本
catch
    [A, R] = geotiffread(fp);    % 老版本
    if isnumeric(R)
        R = refmatToGeoRasterReference(R, size(A));
    end
end

% 确保是 (rows, cols, bands)
if ndims(A) == 2
    A = reshape(A, size(A,1), size(A,2), 1);
end
end

function [latPix, lonPix] = pixel_center_latlon(R, nRows, nCols)
% 用 GeoReference 计算像元中心经纬度
% 对于 GeographicCellsReference，中心点计算：
%   lat = north - (row-0.5)*dLat
%   lon = west  + (col-0.5)*dLon

if ~isprop(R, "LatitudeLimits") || ~isprop(R, "LongitudeLimits")
    error("R 不包含 LatitudeLimits/LongitudeLimits，无法计算经纬度中心。");
end

latLim = double(R.LatitudeLimits);   % [south north]
lonLim = double(R.LongitudeLimits);  % [west east]

south = latLim(1);
north = latLim(2);
west  = lonLim(1);
east  = lonLim(2);

dLat = (north - south) / double(nRows);
dLon = (east  - west ) / double(nCols);

row = (1:nRows)';
col = (1:nCols);

latCenters = north - (row - 0.5) * dLat;  % nRows x 1
lonCenters = west  + (col - 0.5) * dLon;  % 1 x nCols

[lonPix, latPix] = meshgrid(lonCenters, latCenters);

latPix = single(latPix);
lonPix = single(lonPix);
end

function H = fill_nan_only(H, linTarget, tileBand)
% 将 tileBand 的值写入 H，但只写入 H 中仍为 NaN 的位置
v = tileBand(:);
valid = ~isnan(v);     % tileBand 有效值

tidx = linTarget(valid);
v = v(valid);

need = isnan(H(tidx)); % 目标还没填过的
H(tidx(need)) = v(need);
end
