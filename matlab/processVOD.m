%% MCCA AMSR(-E/2) VOD Asc: MAT -> H5 (0.1deg 1800x3600)
% Input:  G:\data\VOD\mat\kuxcVOD\ASC
% Naming:
%   YYYY < 2012 : MCCA_AMSRE_010D_CCXH_VSM_VOD_Asc_YYYYMMDD_V0.nc4.mat
%   YYYY >=2012 : MCCA_AMSR2_010D_CCXH_VSM_VOD_Asc_YYYYMMDD_V0.nc4.mat
% Output: G:\data\VOD\AMSR-VOD\MCCA-VOD(KuCX)\Asc\01h5\YYYYMMDD.h5
% Vars (no processing): c_vod_H, c_vod_V, x_vod_H, x_vod_V, ku_vod_H, ku_vod_V, QC, SM
clc; clear;

inRoot  = "G:\data\VOD\mat\kuxcVOD\ASC";
outRoot = "G:\data\VOD\AMSR-VOD\MCCA-VOD(KuCX)\Asc\01h5";
if ~exist(outRoot, "dir"); mkdir(outRoot); end

% fail log
failLog = fullfile(outRoot, "failed_files.txt");
fidFail = fopen(failLog, "a");
if fidFail < 0
    error("Cannot open fail log: %s", failLog);
end

varsNeed = ["c_vod_H","c_vod_V","x_vod_H","x_vod_V","ku_vod_H","ku_vod_V","QC","SM"];

% list all mat files (both AMSRE / AMSR2)
files = [ ...
    dir(fullfile(inRoot, "MCCA_AMSRE_010D_CCXH_VSM_VOD_Asc_*_V0.nc4.mat")); ...
    dir(fullfile(inRoot, "MCCA_AMSR2_010D_CCXH_VSM_VOD_Asc_*_V0.nc4.mat"))  ...
];

% dedupe
if ~isempty(files)
    [~, ia] = unique(fullfile({files.folder}, {files.name}));
    files = files(ia);
end

if isempty(files)
    fprintf("[ERROR] No MAT matched in %s\n", inRoot);
    fclose(fidFail);
    return
end

% sort by name (stable)
[~, ix] = sort({files.name});
files = files(ix);

for i = 1:numel(files)

    inFile = fullfile(files(i).folder, files(i).name);

    % parse YYYYMMDD (8 digits before _V0.nc4.mat)
    tok = regexp(files(i).name, '_(\d{8})_V0\.nc4\.mat$', 'tokens', 'once');
    if isempty(tok)
        fprintf("[Skip] Cannot parse date: %s\n", files(i).name);
        continue
    end
    YYYYMMDD = string(tok{1});
    YYYY = str2double(extractBetween(YYYYMMDD, 1, 4));

    % sanity check name rule by year (optional warning only)
    isAMSRE = startsWith(files(i).name, "MCCA_AMSRE_");
    isAMSR2 = startsWith(files(i).name, "MCCA_AMSR2_");
    if YYYY < 2012 && ~isAMSRE
        fprintf("[WARN] Year<2012 but file not AMSRE: %s\n", files(i).name);
    elseif YYYY >= 2012 && ~isAMSR2
        fprintf("[WARN] Year>=2012 but file not AMSR2: %s\n", files(i).name);
    end

    outFile = fullfile(outRoot, YYYYMMDD + ".h5");

    % resume: skip if already written
    if exist(outFile, "file") == 2
        fprintf("[Skip done] %s\n", outFile);
        continue
    end

    fprintf("-> %s  (%s)\n", YYYYMMDD, files(i).name);

    try
        S = load(inFile);

        % verify all vars exist
        missing = varsNeed(~isfield(S, cellstr(varsNeed)));
        if ~isempty(missing)
            error("Missing variables in MAT: %s", strjoin(missing, ", "));
        end

        % create output (must create before writeatt)
        if exist(outFile, "file"); delete(outFile); end

        % write each variable as dataset /<var>
        for v = 1:numel(varsNeed)
            vn = varsNeed(v);
            A = S.(vn);

            if ~(isnumeric(A) || islogical(A))
                error("Variable %s is not numeric/logical.", vn);
            end

            % ensure 2D 1800x3600 (no processing, but basic check)
            if ~isequal(size(A), [1800 3600])
                warning("Var %s size is %s (expected 1800x3600). Still writing as-is.", vn, mat2str(size(A)));
            end

            dset = "/" + vn;

            [dtype, chunk] = infer_h5_dtype_and_chunk(A);

            h5create(outFile, dset, size(A), ...
                "Datatype", dtype, ...
                "ChunkSize", chunk, ...
                "Deflate", 4);

            h5write(outFile, dset, A);

            % dataset attribute (optional)
            h5writeatt(outFile, dset, "Name", char(vn));
        end

        % root attributes
        h5writeatt(outFile, "/", "Product", "MCCA AMSR(-E/2) VOD (KuCX) Asc");
        h5writeatt(outFile, "/", "Date", char(YYYYMMDD));
        h5writeatt(outFile, "/", "SourceMat", files(i).name);
        if isAMSRE
            h5writeatt(outFile, "/", "Sensor", "AMSRE");
        elseif isAMSR2
            h5writeatt(outFile, "/", "Sensor", "AMSR2");
        else
            h5writeatt(outFile, "/", "Sensor", "UNKNOWN");
        end

    catch ME
        fprintf("[Skip] Failed: %s\n", ME.message);
        fprintf(fidFail, "%s\t%s\t%s\n", char(YYYYMMDD), inFile, ME.message);
        if exist(outFile, "file"); delete(outFile); end
        continue
    end
end

fclose(fidFail);
disp("Done.");

%% ===== helper: infer dtype and chunk =====
function [dtype, chunk] = infer_h5_dtype_and_chunk(data)
    % keep original class
    dtype = class(data);

    sz = size(data);

    % chunk: safe and not too big
    if numel(sz) == 2
        chunk = [min(180, sz(1)), min(360, sz(2))];
        chunk = max(chunk, [1 1]);
    else
        chunk = ones(1, numel(sz));
        for k = 1:numel(sz)
            chunk(k) = max(1, min(50, sz(k)));
        end
    end
end
