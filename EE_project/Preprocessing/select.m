clc; clear;

%% ============================================================
%  Mode setting
% ============================================================

% Choose one:
% "simulation" -> crop_A + crop_S
% "real"       -> crop_A_r + crop_S_r
mode = "real";

if mode == "simulation"
    folder_A = 'crop_A';
    folder_S = 'crop_S';
    progress_file = 'bad_patches_sim.mat';
elseif mode == "real"
    folder_A = 'crop_A_r';
    folder_S = 'crop_S_r';
    progress_file = 'bad_patches_real.mat';
else
    error('Unknown mode: %s', mode);
end

fprintf('Mode: %s\n', mode);
fprintf('AVIRIS folder: %s\n', folder_A);
fprintf('S2 folder: %s\n', folder_S);
fprintf('Progress file: %s\n', progress_file);

%% ============================================================
%  Load file list
% ============================================================

files_A = dir(fullfile(folder_A, 'patch_*.mat'));
files_S = dir(fullfile(folder_S, 'patch_*.mat'));

names_A = string({files_A.name});
names_S = string({files_S.name});

common_names = intersect(names_A, names_S);
common_names = sort(common_names);

assert(~isempty(common_names), '找不到 A/S 共同 patch 檔案');
assert(length(common_names) == length(names_A), '有 AVIRIS patch 不存在於 S2 folder');
assert(length(common_names) == length(names_S), '有 S2 patch 不存在於 AVIRIS folder');

N = length(common_names);

fprintf('Total common patches: %d\n', N);

%% ============================================================
%  Initialize progress
% ============================================================
bad_files = strings(0,1);
checked_files = strings(0,1);
history_files = strings(0,1);

if isfile(progress_file)
    load(progress_file, 'bad_files', 'checked_files', 'history_files');
    fprintf('載入之前進度：%s\n', progress_file);
end

i = 1;

%% ============================================================
%  Main loop
% ============================================================

while i <= N
    fname = common_names(i);

    % Skip checked file
    if any(checked_files == fname)
        i = i + 1;
        continue;
    end
    %% ====== Read data ======
    A = load(fullfile(folder_A, fname));
    S = load(fullfile(folder_S, fname));

    if isfield(A, 'patch_A')
        patch_A = A.patch_A;
    else
        error('AVIRIS file does not contain patch_A: %s', fname);
    end

    if isfield(S, 'patch_S')
        patch_S = S.patch_S;
    else
        error('S2 file does not contain patch_S: %s', fname);
    end

    %% ====== RGB ======
    % AVIRIS true-color-like display bands
    rgb_A_raw = cat(3, patch_A(:,:,23), patch_A(:,:,12), patch_A(:,:,5));
    
    % Sentinel-2 RGB: B4, B3, B2
    % Assuming patch_S band order is 12-band Sentinel-2 without B10:
    % [B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12]
    rgb_S_raw = cat(3, patch_S(:,:,4), patch_S(:,:,3), patch_S(:,:,2));
    
    % Raw RGB statistics before display normalization
    rgbA_raw_mean = mean(rgb_A_raw(:), 'omitnan');
    rgbS_raw_mean = mean(rgb_S_raw(:), 'omitnan');
    rgb_mean_ratio = rgbA_raw_mean / (rgbS_raw_mean + 1e-8);
    
    % Display only
    if mode == "real"
        [rgb_A, rgb_S] = normalize_by_S2_scale(rgb_A_raw, rgb_S_raw);
    else
        rgb_A = normalize_img(rgb_A_raw);
        rgb_S = normalize_img(rgb_S_raw);
    end

    %% ====== Statistics ======
    A_max  = max(patch_A(:));
    A_mean = mean(patch_A(:), 'omitnan');
    A_std  = std(double(patch_A(:)), 'omitnan');

    S_max  = max(patch_S(:));
    S_mean = mean(patch_S(:), 'omitnan');
    S_std  = std(double(patch_S(:)), 'omitnan');

    AS_mean_ratio = A_mean / (S_mean + 1e-8);
    AS_max_ratio = A_max / (S_max + 1e-8);

    zero_A = nnz(abs(patch_A) <= 1e-6) / numel(patch_A);
    zero_S = nnz(abs(patch_S) <= 1e-6) / numel(patch_S);

    nan_A = nnz(~isfinite(patch_A)) / numel(patch_A);
    nan_S = nnz(~isfinite(patch_S)) / numel(patch_S);

    rgb_corr = NaN;
    if mode == "real"
        vA = double(rgb_A(:));
        vS = double(rgb_S(:));
        valid = isfinite(vA) & isfinite(vS);
    
        if nnz(valid) > 10 && std(vA(valid)) > 1e-8 && std(vS(valid)) > 1e-8
            rgb_corr = corr(vA(valid), vS(valid));
        end
    end

    %% ====== Print info ======
    fprintf('\n==============================\n');
    fprintf('mode = %s\n', mode);
    fprintf('idx = %d / %d\n', i, N);
    fprintf('File: %s\n', fname);
    fprintf('A max / mean / std = %.6f / %.6f / %.6f\n', A_max, A_mean, A_std);
    fprintf('S max / mean / std = %.6f / %.6f / %.6f\n', S_max, S_mean, S_std);
    fprintf('zero_A = %.4f, zero_S = %.4f\n', zero_A, zero_S);
    fprintf('nan_A  = %.4f, nan_S  = %.4f\n', nan_A, nan_S);
    fprintf('RGB raw mean ratio A/S = %.4f\n', rgb_mean_ratio);
    fprintf('A/S mean ratio = %.4f, A/S max ratio = %.4f\n', AS_mean_ratio, AS_max_ratio);
    if mode == "real"
        fprintf('real S2 RGB corr = %.4f  (建議 > 0.70)\n', rgb_corr);
    end
    fprintf('判斷參考：優先看 A/S 是否對齊、NaN/Inf、黑邊/nodata。\n');
    fprintf('城市/水體 patch 的 A/S mean ratio 可當 warning，不建議直接當刪除標準。\n');
    fprintf('若 A/S ratio 極低但 corr 高、結構清楚、無黑邊，通常可保留。\n');

    %% ====== Display ======
    figure(1); clf;
    set(gcf, 'Position', [100,300,1500,600]);

    subplot(1,3,1);
    imshow(rgb_A);
    title(sprintf('AVIRIS (%s)', fname), 'Interpreter', 'none');

    subplot(1,3,2);
    imshow(rgb_S);
    title(sprintf('Sentinel-2 (%s)', mode), 'Interpreter', 'none');

    subplot(1,3,3);
    axis off;

    info_text = {
        sprintf('mode = %s', mode)
        sprintf('idx = %d / %d', i, N)
        sprintf('file: %s', fname)
        ''
        '--- AVIRIS patch_A ---'
        sprintf('max  = %.6f', A_max)
        sprintf('mean = %.6f', A_mean)
        sprintf('zero = %.4f', zero_A)
        ''
        '--- Sentinel-2 patch_S ---'
        sprintf('max  = %.6f', S_max)
        sprintf('mean = %.6f', S_mean)
        sprintf('zero = %.4f', zero_S)
        ''
        '--- Raw scale check ---'
        sprintf('A/S mean ratio = %.4f', AS_mean_ratio)
        sprintf('A/S max ratio  = %.4f', AS_max_ratio)
        sprintf('RGB raw ratio  = %.4f', rgb_mean_ratio)
        ''
        '--- Display RGB ---'
        sprintf('real RGB corr = %.4f', rgb_corr)
    };

    text(0, 1, info_text, ...
        'Units', 'normalized', ...
        'VerticalAlignment', 'top', ...
        'FontName', 'Consolas', ...
        'FontSize', 10);

    sgtitle('k=keep | d=delete | b=back | q=quit');

    %% ====== Keyboard control ======
    waitforbuttonpress;
    key = get(gcf, 'CurrentCharacter');

    if key == 'd'
        bad_files(end+1,1) = fname;
        checked_files(end+1,1) = fname;
        history_files(end+1,1) = fname;
        fprintf('❌ bad: %s\n', fname);
        i = i + 1;

    elseif key == 'k'
        checked_files(end+1,1) = fname;
        history_files(end+1,1) = fname;
        fprintf('✔ keep: %s\n', fname);
        i = i + 1;

    elseif key == 'b'
        if ~isempty(history_files)
            prev = history_files(end);
            history_files(end) = [];

            checked_files(checked_files == prev) = [];
            bad_files(bad_files == prev) = [];

            i = find(common_names == prev, 1);
            fprintf('🔙 回到 %s\n', prev);
        else
            fprintf('⚠️ 沒有上一張\n');
        end

    elseif key == 'q'
        fprintf('⛔ 手動中止\n');
        save(progress_file, 'bad_files', 'checked_files', 'history_files', 'mode');
        return;
    end

    %% ====== Save progress every 10 operations ======
    if mod(length(history_files), 10) == 0
        save(progress_file, 'bad_files', 'checked_files', 'history_files', 'mode');
        fprintf('💾 已儲存進度\n');
    end
end


save(progress_file, 'bad_files', 'checked_files', 'history_files', 'mode');

fprintf('\n完成！壞資料數量: %d\n', length(bad_files));
fprintf('Progress saved to: %s\n', progress_file);

function img = normalize_img(img)

    img = single(img);
    img(~isfinite(img)) = 0;

    for ch = 1:size(img,3)
        band = img(:,:,ch);

        low  = prctile(band(:), 1);
        high = prctile(band(:), 99);

        band = (band - low) / (high - low + 1e-8);

        band(band < 0) = 0;
        band(band > 1) = 1;

        img(:,:,ch) = band;
    end
end
function [imgA, imgS] = normalize_by_S2_scale(imgA_raw, imgS_raw)

    imgA_raw = single(imgA_raw);
    imgS_raw = single(imgS_raw);

    imgA_raw(~isfinite(imgA_raw)) = 0;
    imgS_raw(~isfinite(imgS_raw)) = 0;

    % Use Sentinel-2 display scale
    low  = prctile(imgS_raw(:), 1);
    high = prctile(imgS_raw(:), 99);

    imgA = (imgA_raw - low) / (high - low + 1e-8);
    imgS = (imgS_raw - low) / (high - low + 1e-8);

    imgA(imgA < 0) = 0;
    imgA(imgA > 1) = 1;

    imgS(imgS < 0) = 0;
    imgS(imgS > 1) = 1;
end