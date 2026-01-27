%% Plot processed LST global map (0.1deg) for 20020711
clc; clear;

h5file = "G:\data\LST_yupei\01h5\20020711.h5";
dset   = "/LST_Day";

if exist(h5file, "file") ~= 2
    error("File not found: %s", h5file);
end

% read
L = h5read(h5file, dset);
L = double(L);

fprintf("Read %s\n", h5file);
fprintf("Dataset: %s, size=%dx%d\n", dset, size(L,1), size(L,2));

% target grid
lat = 89.95:-0.1:-89.95;      % 1800
lon = -179.95:0.1:179.95;     % 3600

% fix orientation to [lat, lon]
if isequal(size(L), [numel(lat), numel(lon)])
    % ok
elseif isequal(size(L), [numel(lon), numel(lat)])
    warning("Data seems [lon,lat]. Transpose to [lat,lon].");
    L = L.';
else
    warning("Unexpected size. Plotting anyway (axes may be off).");
end

% quick sanity stats
nValid = sum(~isnan(L(:)));
fprintf("Valid pixels: %d / %d\n", nValid, numel(L));
if nValid > 0
    fprintf("Min/Max (omitnan): %.2f / %.2f\n", min(L(:),[],'omitnan'), max(L(:),[],'omitnan'));
end

% plot
figure("Color","w");
imagesc(lon, lat, L, "AlphaData", ~isnan(L));
set(gca, "YDir", "normal");
axis tight;
xlabel("Longitude");
ylabel("Latitude");
title("LST Day (K) - 20020711 (0.1°)");
colorbar;

% 可选：限定色标范围（K）
% caxis([220 330]);
