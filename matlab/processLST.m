%% LST_yupei: 0.05° (3600x7200) -> 0.1° (1800x3600), daily YYYYMMDD.h5
% Output only one dataset: /LST_Day (double, K), NO QC screening.
% Robust: blockwise h5read to avoid crash; failures are logged and skipped.
clc; clear;

inRoot  = "G:\data\LST_yupei\005";
outRoot = "G:\data\LST_yupei\01h5";
if ~exist(outRoot, 'dir'); mkdir(outRoot); end

% log for failed files
failLog = fullfile(outRoot, "failed_files.txt");
fidFail = fopen(failLog, "a");
if fidFail < 0, error("Cannot open fail log: %s", failLog); end

% ========= Default attributes (fallback if file lacks attrs) =========
default_scaleFactor = 0.02;
default_addOffset   = 0.0;
default_fillValue   = 0;
default_validMin    = 7500;
default_validMax    = 65535;

% ========= list files (兼容 MOD/MYD + All-weather/Clear-sky) =========
files = [ ...
    dir(fullfile(inRoot, "MOD11C1_*_All-weather.h5")); ...
    dir(fullfile(inRoot, "MYD11C1_*_All-weather.h5")); ...
    dir(fullfile(inRoot, "MOD11C1_*_Clear-sky.h5"));   ...
    dir(fullfile(inRoot, "MYD11C1_*_Clear-sky.h5"))    ...
];

% 去重（如果有重复）
if ~isempty(files)
    [~, ia] = unique(fullfile({files.folder}, {files.name}));
    files = files(ia);
end

if isempty(files)
    fprintf("[ERROR] No input found under %s\n", inRoot);
    fclose(fidFail);
    return
end

[~, ix] = sort({files.name});
files = files(ix);

for i = 1:numel(files)

    inFile = fullfile(files(i).folder, files(i).name);

    % ---- parse YYYY + DOY from filename ----
    tok = regexp(files(i).name, '^(MOD11C1|MYD11C1)_(\d{4})(\d{3})_(All-weather|Clear-sky)\.h5$', 'tokens', 'once');
    if isempty(tok)
        fprintf("[Skip] Bad name: %s\n", files(i).name);
        continue
    end
    YYYY = str2double(tok{2});
    DOY  = str2double(tok{3});

    dt = datenum(YYYY,1,1) + DOY - 1;
    YYYYMMDD = string(datestr(dt, "yyyymmdd"));

    outFile = fullfile(outRoot, YYYYMMDD + ".h5");

    % resume: skip done
    if exist(outFile, "file")
        fprintf("[Skip done] %s\n", outFile);
        continue
    end

    fprintf("-> %s  (%s)\n", YYYYMMDD, files(i).name);

    try
        % ---- find LST dataset robustly (may be in groups) ----
        lstPathCandidates = { ...
            "/LST_Day_CMG","/LST_Day","/LST_Day_Clear","/LST_Day_clear", ...
            "LST_Day_CMG","LST_Day","LST_Day_Clear","LST_Day_clear" ...
        };

        lstDset = find_dataset_path(inFile, lstPathCandidates);

        if strlength(lstDset) == 0
            error("Cannot find LST dataset in file.");
        end

        % ---- get dataset size ----
        infoL = h5info(inFile, char(lstDset));
        dims  = infoL.Dataspace.Size;  % e.g., [3600 7200]
        if numel(dims) ~= 2
            error("Unexpected LST rank: %d", numel(dims));
        end
        nR = dims(1); nC = dims(2);

        if mod(nR,2)~=0 || mod(nC,2)~=0
            warning("LST dims not even: %dx%d, will truncate to even.", nR, nC);
        end
        nR2 = floor(nR/2)*2;
        nC2 = floor(nC/2)*2;

        outR = nR2/2;
        outC = nC2/2;

        % ---- read attributes if available (fallback to defaults) ----
        [scaleFactor, addOffset, fillValue, validMin, validMax] = ...
            get_lst_attrs_safe(inFile, lstDset, ...
                               default_scaleFactor, default_addOffset, default_fillValue, default_validMin, default_validMax);

        % ---- preallocate output (0.1°) ----
        lst_01 = nan(outR, outC, 'double');

        % ---- blockwise read rows (must be even) ----
        blkR = 200;
        blkR = blkR - mod(blkR,2);
        if blkR < 2, blkR = 2; end

        r0 = 1;
        while r0 <= nR2

            rCount = min(blkR, nR2 - r0 + 1);
            rCount = rCount - mod(rCount,2);
            if rCount <= 0, break; end

            % read LST block
            lstBlk = h5read(inFile, char(lstDset), [r0 1], [rCount nC2]);
            lstBlk = double(lstBlk);

            % ---- fill -> NaN ----
            lstBlk(lstBlk == fillValue) = NaN;

            % ---- raw validity range (保留原有逻辑：超出范围设 NaN) ----
            % 如果你希望“只处理 fill，不做 valid_range 过滤”，注释掉下面这行即可：
            lstBlk(lstBlk < validMin | lstBlk > validMax) = NaN;

            % ---- scale to K ----
            lstBlk = lstBlk * scaleFactor + addOffset;

            % ---- 2x2 valid mean (no mean()) ----
            outBlk = blockMean2x2_valid_noMean(lstBlk);

            outRowStart = (r0-1)/2 + 1;
            outRowEnd   = outRowStart + size(outBlk,1) - 1;
            lst_01(outRowStart:outRowEnd, :) = outBlk;

            r0 = r0 + rCount;

            clear lstBlk outBlk
            drawnow;
        end

        % ---- write output h5 (only /LST_Day) ----
        if exist(outFile, "file"); delete(outFile); end

        h5create(outFile, "/LST_Day", size(lst_01), ...
            "Datatype","double", "ChunkSize",[180 360], "Deflate",4);
        h5write(outFile, "/LST_Day", lst_01);

        % attributes (optional)
        h5writeatt(outFile, "/", "Product", "LST_yupei (no QC, aggregated 0.05->0.1)");
        h5writeatt(outFile, "/", "Units", "K");
        h5writeatt(outFile, "/", "ScaleFactorUsed", scaleFactor);
        h5writeatt(outFile, "/", "AddOffsetUsed", addOffset);
        h5writeatt(outFile, "/", "FillValueRaw", fillValue);
        h5writeatt(outFile, "/", "RawValidRangeUsed", sprintf("%g-%g (before scaling)", validMin, validMax));
        h5writeatt(outFile, "/", "Resolution", "0.1 degree (2x2 valid-mean from 0.05 degree)");
        h5writeatt(outFile, "/", "SourceFile", files(i).name);
        h5writeatt(outFile, "/", "InputLSTDataset", char(lstDset));
        h5writeatt(outFile, "/", "InputProduct", tok{1});
        h5writeatt(outFile, "/", "InputSkyTag",  tok{4});

        clear lst_01

    catch ME
        fprintf("[Skip] Failed: %s\n", ME.message);
        fprintf(fidFail, "%s\t%s\t%s\n", char(YYYYMMDD), inFile, ME.message);
        if exist(outFile, "file"); delete(outFile); end
        continue
    end
end

fclose(fidFail);
disp("Done.");

%% ===================== helper functions =====================

function dset = find_dataset_path(h5file, candidates)
% Find the first existing dataset path among candidates.
% If candidate has no leading '/', we will try both with and without it.
    dset = "";
    try
        root = h5info(h5file);
    catch
        return
    end

    allPaths = list_all_datasets(root, "");

    for k = 1:numel(candidates)
        c = string(candidates{k});
        if startsWith(c, "/")
            candsToTry = [c, extractAfter(c,1)];
        else
            candsToTry = ["/"+c, c];
        end
        for t = 1:numel(candsToTry)
            if any(strcmp(allPaths, candsToTry(t)))
                dset = candsToTry(t);
                return
            end
        end
    end
end

function paths = list_all_datasets(node, prefix)
% Recursively list dataset full paths under an h5info node
    paths = strings(0,1);

    if isfield(node, "Datasets") && ~isempty(node.Datasets)
        for i = 1:numel(node.Datasets)
            nm = string(node.Datasets(i).Name);
            if prefix == ""
                paths(end+1,1) = "/" + nm; %#ok<AGROW>
            else
                paths(end+1,1) = prefix + "/" + nm; %#ok<AGROW>
            end
        end
    end

    if isfield(node, "Groups") && ~isempty(node.Groups)
        for g = 1:numel(node.Groups)
            grp = node.Groups(g);
            grpName = string(grp.Name);
            paths = [paths; list_all_datasets(grp, grpName)]; %#ok<AGROW>
        end
    end
end

function [scaleFactor, addOffset, fillValue, validMin, validMax] = ...
    get_lst_attrs_safe(h5file, dset, dScale, dOffset, dFill, dVmin, dVmax)
% Try reading common attribute names; if absent use defaults.
    scaleFactor = dScale;
    addOffset   = dOffset;
    fillValue   = dFill;
    validMin    = dVmin;
    validMax    = dVmax;

    function v = try_att(attrName)
        v = [];
        try
            v = h5readatt(h5file, char(dset), attrName);
        catch
        end
    end

    v = try_att("scale_factor"); if ~isempty(v), scaleFactor = double(v); end
    v = try_att("Scale factor"); if ~isempty(v), scaleFactor = double(v); end

    v = try_att("add_offset");   if ~isempty(v), addOffset   = double(v); end
    v = try_att("Add offset");   if ~isempty(v), addOffset   = double(v); end

    v = try_att("_FillValue");   if ~isempty(v), fillValue   = double(v); end
    v = try_att("Fill Value");   if ~isempty(v), fillValue   = double(v); end

    v = try_att("valid_range");
    if ~isempty(v) && numel(v) >= 2
        validMin = double(v(1));
        validMax = double(v(2));
    end
end

function out = blockMean2x2_valid_noMean(A)
% 2x2 valid-mean without mean()
% A: (2M x 2N) double with NaN invalid
    r = size(A,1); c = size(A,2);
    r2 = floor(r/2)*2;
    c2 = floor(c/2)*2;
    A = A(1:r2, 1:c2);

    a11 = A(1:2:end, 1:2:end);
    a21 = A(2:2:end, 1:2:end);
    a12 = A(1:2:end, 2:2:end);
    a22 = A(2:2:end, 2:2:end);

    v11 = ~isnan(a11); v21 = ~isnan(a21); v12 = ~isnan(a12); v22 = ~isnan(a22);

    a11(~v11) = 0; a21(~v21) = 0; a12(~v12) = 0; a22(~v22) = 0;

    sumv = a11 + a21 + a12 + a22;
    cnt  = double(v11) + double(v21) + double(v12) + double(v22);

    out = sumv ./ cnt;
    out(cnt == 0) = NaN;
end
