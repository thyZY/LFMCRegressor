%% MCCA AMSR(-E/2) VOD Asc: MAT -> H5 (BATCH FIX DATES) + verify
% Fix corrupted daily H5 by regenerating from MAT.
% Input MAT : G:\data\VOD\mat\kuxcVOD\ASC
% Output H5 : G:\data\VOD\AMSR-VOD\MCCA-VOD(KuCX)\Asc\01h5\YYYYMMDD.h5

clc; clear;

inRoot  = "G:\data\VOD\mat\kuxcVOD\ASC";
outRoot = "G:\data\VOD\AMSR-VOD\MCCA-VOD(KuCX)\Asc\01h5";
if ~exist(outRoot, "dir"); mkdir(outRoot); end

% ===== 需要修复的日期（来自你的 Python 扫描结果）=====
datesToFix = [ ...
    "20120708"; ...
    "20130402"; ...
    "20130818"; ...
    "20140517"; ...
    "20170625"; ...
    "20170916"  ...
];

% 固定变量
varsNeed = ["c_vod_H","c_vod_V","x_vod_H","x_vod_V","ku_vod_H","ku_vod_V","QC","SM"];

% 记录日志
logFile = fullfile(outRoot, "rebuild_bad_h5_log.txt");
fid = fopen(logFile, "a");
if fid < 0
    error("Cannot open log file: %s", logFile);
end
fprintf(fid, "\n===== Rebuild run at %s =====\n", datestr(now));

% 验证参数（可调）
doFullRead = false;          % true=对每个变量都 full read（慢）；建议 false
chunkProbeMax = 500;         % 每个变量最多 probe 多少个chunk（足够定位坏块；设 Inf 就全扫，慢）

for k = 1:numel(datesToFix)
    YYYYMMDD = datesToFix(k);
    YYYY = str2double(extractBetween(YYYYMMDD, 1, 4));

    fprintf("\n================================================================================\n");
    fprintf("[FIX] %s\n", YYYYMMDD);
    fprintf(fid, "[FIX] %s\n", YYYYMMDD);

    % ----- 找对应 MAT（主规则 + fallback）-----
    patAMSRE = "MCCA_AMSRE_010D_CCXH_VSM_VOD_Asc_" + YYYYMMDD + "_V0.nc4.mat";
    patAMSR2 = "MCCA_AMSR2_010D_CCXH_VSM_VOD_Asc_" + YYYYMMDD + "_V0.nc4.mat";

    pAMSRE = fullfile(inRoot, patAMSRE);
    pAMSR2 = fullfile(inRoot, patAMSR2);

    inFile = "";
    sensor = "UNKNOWN";

    % 按年份优先
    if YYYY < 2012
        if exist(pAMSRE, "file") == 2
            inFile = pAMSRE; sensor = "AMSRE";
        elseif exist(pAMSR2, "file") == 2
            inFile = pAMSR2; sensor = "AMSR2";
        end
    else
        if exist(pAMSR2, "file") == 2
            inFile = pAMSR2; sensor = "AMSR2";
        elseif exist(pAMSRE, "file") == 2
            inFile = pAMSRE; sensor = "AMSRE";
        end
    end

    if inFile == ""
        msg = sprintf("[ERROR] MAT not found for %s", YYYYMMDD);
        fprintf("%s\n", msg);
        fprintf(fid, "%s\n", msg);
        continue
    end

    outFile = fullfile(outRoot, YYYYMMDD + ".h5");
    fprintf("Input MAT : %s\n", inFile);
    fprintf("Output H5 : %s\n", outFile);

    % ----- 重建（删除旧文件）-----
    try
        if exist(outFile, "file") == 2
            delete(outFile);
        end

        S = load(inFile);

        missing = varsNeed(~isfield(S, cellstr(varsNeed)));
        if ~isempty(missing)
            error("Missing variables in MAT: %s", strjoin(missing, ", "));
        end

        % 写入每个变量
        for v = 1:numel(varsNeed)
            vn = varsNeed(v);
            A = S.(vn);

            if ~(isnumeric(A) || islogical(A))
                error("Variable %s is not numeric/logical.", vn);
            end

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
            h5writeatt(outFile, dset, "Name", char(vn));

            fprintf("[WRITE OK] %s  dtype=%s  size=%s  chunk=%s\n", ...
                dset, dtype, mat2str(size(A)), mat2str(chunk));
        end

        % root attrs
        h5writeatt(outFile, "/", "Product", "MCCA AMSR(-E/2) VOD (KuCX) Asc");
        h5writeatt(outFile, "/", "Date", char(YYYYMMDD));
        [~, srcName, srcExt] = fileparts(inFile);
        h5writeatt(outFile, "/", "SourceMat", srcName + srcExt);
        h5writeatt(outFile, "/", "Sensor", sensor);

        fprintf("[ATTR OK] root attributes written.\n");

    catch ME
        msg = sprintf("[REBUILD FAIL] %s  err=%s", YYYYMMDD, ME.message);
        fprintf("%s\n", msg);
        fprintf(fid, "%s\n", msg);
        if exist(outFile, "file") == 2
            delete(outFile);
        end
        continue
    end

    % ----- 写后验证：对每个变量 probe + 可选 full read -----
    fprintf("\n================ VERIFY %s ================\n", YYYYMMDD);
    for v = 1:numel(varsNeed)
        ds = "/" + varsNeed(v);
        fprintf("\n[VERIFY] %s\n", ds);

        try
            dinfo = h5info(outFile, ds);
            sz = dinfo.Dataspace.Size;
            chunk = dinfo.ChunkSize;

            % small read (1,1)
            try
                tmp = h5read(outFile, ds, [1 1], [min(8,sz(1)) min(8,sz(2))]); %#ok<NASGU>
                clear tmp
                fprintf("  [OK] small read (1,1)\n");
            catch ME2
                fprintf("  [FAIL] small read (1,1): %s\n", ME2.message);
            end

            % small read (center)
            try
                s1 = max(1, floor(sz(1)/2));
                s2 = max(1, floor(sz(2)/2));
                tmp = h5read(outFile, ds, [s1 s2], [min(8,sz(1)-s1+1) min(8,sz(2)-s2+1)]); %#ok<NASGU>
                clear tmp
                fprintf("  [OK] small read (center)\n");
            catch ME2
                fprintf("  [FAIL] small read (center): %s\n", ME2.message);
            end

            % chunk probe（找第一个坏块即可）
            if isempty(chunk)
                fprintf("  [INFO] ChunkSize empty (not chunked)\n");
            else
                badFound = false;
                nProbe = 0;

                for i = 1:chunk(1):sz(1)
                    for j = 1:chunk(2):sz(2)
                        count1 = min(chunk(1), sz(1)-i+1);
                        count2 = min(chunk(2), sz(2)-j+1);

                        try
                            tmp = h5read(outFile, ds, [i j], [count1 count2]); %#ok<NASGU>
                            clear tmp
                        catch ME2
                            fprintf("  [BAD CHUNK] start=(%d,%d) count=(%d,%d) err=%s\n", ...
                                i, j, count1, count2, ME2.message);
                            fprintf(fid, "  [BAD CHUNK] %s %s start=(%d,%d) err=%s\n", ...
                                YYYYMMDD, ds, i, j, ME2.message);
                            badFound = true;
                            break
                        end

                        nProbe = nProbe + 1;
                        if isfinite(chunkProbeMax) && nProbe >= chunkProbeMax
                            break
                        end
                    end
                    if badFound, break; end
                    if isfinite(chunkProbeMax) && nProbe >= chunkProbeMax
                        break
                    end
                end

                if ~badFound
                    fprintf("  [OK] chunk probe passed (probed %d chunks)\n", nProbe);
                end
            end

            % 可选 full read（不建议每个变量都开）
            if doFullRead
                try
                    tmp = h5read(outFile, ds); %#ok<NASGU>
                    clear tmp
                    fprintf("  [OK] full read\n");
                catch ME2
                    fprintf("  [FAIL] full read: %s\n", ME2.message);
                    fprintf(fid, "  [FULL READ FAIL] %s %s err=%s\n", YYYYMMDD, ds, ME2.message);
                end
            end

        catch ME
            fprintf("  [VERIFY FAIL] %s\n", ME.message);
            fprintf(fid, "  [VERIFY FAIL] %s %s err=%s\n", YYYYMMDD, ds, ME.message);
        end
    end

    fprintf("\n[FIX DONE] %s\n", YYYYMMDD);
    fprintf(fid, "[FIX DONE] %s\n", YYYYMMDD);
end

fclose(fid);
disp("All done.");

%% ===== helper: infer dtype and chunk =====
function [dtype, chunk] = infer_h5_dtype_and_chunk(data)
    dtype = class(data);
    sz = size(data);

    if numel(sz) == 2
        % 保持与你原始写法一致（180x360）
        chunk = [min(180, sz(1)), min(360, sz(2))];
        chunk = max(chunk, [1 1]);
    else
        chunk = ones(1, numel(sz));
        for k = 1:numel(sz)
            chunk(k) = max(1, min(50, sz(k)));
        end
    end
end
