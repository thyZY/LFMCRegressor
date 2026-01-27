function diagnose_one_LST_hdf()
% Diagnose MODIS LST HDF (the day after last success)
clc;

inRoot   = "G:\data\MOD11C1 CMG LST\005D";
lastOkH5 = "20161214";   % last successfully generated h5 date (yyyymmdd)

targetDate = datestr(datenum(lastOkH5,"yyyymmdd")+1, "yyyymmdd");
fprintf("Last OK: %s -> Target to diagnose: %s\n", lastOkH5, targetDate);

% Log (so we know where it stops)
logFile = fullfile(inRoot, "diagnose_LST_log.txt");
diary(logFile); diary on;

dt   = datenum(targetDate, "yyyymmdd");
yyyy = str2double(targetDate(1:4));
doy  = dt - datenum(yyyy,1,1) + 1;
fprintf("Target date %s corresponds to YYYY=%d, DOY=%03d\n", targetDate, yyyy, doy);

% Locate HDF file
yearFolder = fullfile(inRoot, sprintf("%04d", yyyy));
if exist(yearFolder, "dir")
    searchDir = yearFolder;
else
    searchDir = inRoot;
end

pattern = sprintf("*A%04d%03d*.hdf", yyyy, doy);
cands = dir(fullfile(searchDir, pattern));
if isempty(cands)
    cands = dir(fullfile(searchDir, "**", pattern));
end

if isempty(cands)
    fprintf("[ERROR] Cannot find HDF matching %s under %s\n", pattern, searchDir);
    diary off;
    return
end

fprintf("Found %d candidate file(s):\n", numel(cands));
for k = 1:numel(cands)
    fprintf("  %d) %s  (%.2f MB)\n", k, fullfile(cands(k).folder, cands(k).name), cands(k).bytes/1024/1024);
end

inFile = fullfile(cands(1).folder, cands(1).name);
fprintf("\n[DIAG] Using file: %s\n", inFile);

% OS-level open test
fid = fopen(inFile, "r");
if fid < 0
    fprintf("[ERROR] OS-level fopen failed (file missing/locked).\n");
    diary off;
    return
else
    fclose(fid);
    fprintf("[OK] OS-level fopen succeeded.\n");
end

% List SDS names (SD interface)
try
    sdsNames = hdf4_list_sds(inFile);
    fprintf("\n[OK] SDS list (%d datasets):\n", numel(sdsNames));
    for k = 1:numel(sdsNames)
        fprintf("  - %s\n", sdsNames{k});
    end
catch ME
    fprintf("\n[ERROR] Failed to list SDS via SD interface:\n%s\n", ME.message);
    fprintf("Next step: try MATLAB hdfinfo/hdfread to see if only SD fails.\n");
end

% Try hdfinfo as a fallback to see structure (may work even if SD listing fails)
try
    info = hdfinfo(inFile);
    fprintf("\n[OK] hdfinfo succeeded. Top-level has %d Vgroups.\n", numel(info.Vgroup));
catch ME
    fprintf("\n[ERROR] hdfinfo also failed:\n%s\n", ME.message);
    fprintf("If hdfinfo fails too, file is likely corrupted or triggers HDF4 crash.\n");
    diary off;
    return
end

% Try reading the known SDS with SD subset read (safer)
fprintf("\n[DIAG] Try reading small subset (200x300) of LST_Day_CMG and QC_Day via SD...\n");
start = [0 0];   % 0-based
edge  = [200 300];
try
    lstSub = hdf4_read_sds_subset(inFile, "LST_Day_CMG", start, edge);
    qcSub  = hdf4_read_sds_subset(inFile, "QC_Day",     start, edge);
    fprintf("[OK] Subset read succeeded.\n");
    fprintf("  LST subset: class=%s size=%s min=%g max=%g\n", class(lstSub), mat2str(size(lstSub)), double(min(lstSub(:))), double(max(lstSub(:))));
    fprintf("  QC  subset: class=%s size=%s min=%g max=%g\n", class(qcSub),  mat2str(size(qcSub)),  double(min(qcSub(:))),  double(max(qcSub(:))));
catch ME
    fprintf("[ERROR] Subset read failed:\n%s\n", ME.message);
    fprintf("This strongly suggests file corruption or SD-interface incompatibility on this file.\n");
    diary off;
    return
end

fprintf("\n[DIAG DONE] If subset works but batch crashes, then full read/decompression is the trigger.\n");
diary off;

end

%% ---------------- local helpers (must be AFTER the main function) ----------------

function names = hdf4_list_sds(filename)
    import matlab.io.hdf4.*
    sdID = sd.start(filename, "read");
    c0 = onCleanup(@() sd.close(sdID));

    [nDatasets, ~] = sd.fileInfo(sdID);
    names = cell(nDatasets, 1);

    for idx = 0:nDatasets-1
        sdsID = sd.select(sdID, idx);
        c1 = onCleanup(@() sd.endAccess(sdsID));
        [name, ~, ~, ~] = sd.getInfo(sdsID);
        names{idx+1} = name;
    end
end

function out = hdf4_read_sds_subset(filename, sdsName, start, edge)
    import matlab.io.hdf4.*
    sdID = sd.start(filename, "read");
    c0 = onCleanup(@() sd.close(sdID));

    idx = sd.nameToIndex(sdID, sdsName);
    sdsID = sd.select(sdID, idx);
    c1 = onCleanup(@() sd.endAccess(sdsID));

    out = sd.readData(sdsID, start, [], edge);
end
