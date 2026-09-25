clc; clear;

clear delete   % 防止覆蓋 MATLAB built-in delete

%% ============================================================
%  Mode setting
% ============================================================
% Choose one:
% "real"       -> delete only from crop_S
% "simulation" -> delete only from crop_S_sim

mode = "real";

if mode == "real"
    folder_A = "crop_A_r";
    folder_S = 'crop_S_r';
    bad_file = 'bad_patches_real.mat';

elseif mode == "simulation"
    folder_A = "crop_A";
    folder_S = 'crop_S';
    bad_file = 'bad_patches_sim.mat';
end

fprintf('Mode: %s\n', mode);
fprintf('Only delete from S2 folder: %s\n', folder_S);
fprintf('Bad patch file: %s\n', bad_file);
fprintf('Note: crop_A will NOT be deleted.\n\n');

%% ============================================================
%  Load bad patch list
% ============================================================

load(bad_file);

% 支援新版 bad_files，也支援舊版 bad_list
if exist('bad_files', 'var')
    delete_files = string(bad_files);

else
    error('在 %s 中找不到 bad_files 或 bad_list。', bad_file);
end

% 統一轉成 string
delete_files = string(delete_files);

% 移除 missing
delete_files = delete_files(~ismissing(delete_files));

% 移除空字串
delete_files = delete_files(strlength(delete_files) > 0);

% 移除前後空白
delete_files = strip(delete_files);

% 再去重複
delete_files = unique(delete_files, 'stable');

fprintf('Files marked as bad after cleaning: %d\n', length(delete_files));

if isempty(delete_files)
    fprintf("沒有需要刪除的檔案。\n");
    return;
end

deleted_S = 0;
deleted_A = 0;
missing_S = 0;
missing_A = 0;

for k = 1:numel(delete_files)

    filename = char(delete_files(k));

    file_S = fullfile(folder_S, filename);
    file_A = fullfile(folder_A, filename);

    % 刪除 Sentinel-2
    if isfile(file_S)
        builtin("delete", file_S);
        deleted_S = deleted_S + 1;
        fprintf("Deleted S2     : %s\n", filename);
    else
        missing_S = missing_S + 1;
        fprintf("Missing S2     : %s\n", filename);
    end

    % 刪除 AVIRIS
    if isfile(file_A)
        builtin("delete", file_A);
        deleted_A = deleted_A + 1;
        fprintf("Deleted AVIRIS : %s\n", filename);
    else
        missing_A = missing_A + 1;
        fprintf("Missing AVIRIS : %s\n", filename);
    end

end


fprintf('完成刪除。\n');
fprintf("Deleted S2 files     : %d\n", deleted_S);
fprintf("Deleted AVIRIS files : %d\n", deleted_A);
fprintf("Missing S2 files     : %d\n", missing_S);
fprintf("Missing AVIRIS files : %d\n", missing_A);

log_file = sprintf('deleted_s2_only_%s.mat', mode);

save(log_file, ...
    "delete_files", ...
    "mode", ...
    "folder_S", ...
    "folder_A", ...
    "deleted_S", ...
    "deleted_A", ...
    "missing_S", ...
    "missing_A");

fprintf('Deletion log saved to: %s\n', log_file);