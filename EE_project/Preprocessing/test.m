clc; clear;

% ==============================
% 抽幾張影像快速檢查
% 顯示用 percentile normalization，避免亮度被壓黑
% 每張圖印簡單檢查：range / mean / std / zero% / NaN% / RGB correlation
% ==============================

folder_A = 'crop_A';
folder_S = 'crop_S';   % real S2 用 crop_S_r；simulation 用 crop_S

ids = [4, 112, 464, 505, 568];

use_shared_scale = true;   % false：各自 normalize，看圖有沒有內容；true：共用 scale，看亮度差異

for k = 1:length(ids)
    patch_id = ids(k);
    file_name = sprintf('patch_%05d.mat', patch_id);

    path_A = fullfile(folder_A, file_name);
    path_S = fullfile(folder_S, file_name);

    if ~isfile(path_A) || ~isfile(path_S)
        warning('缺少配對檔案：%s 或 %s', path_A, path_S);
        continue;
    end

    A = load(path_A);
    S = load(path_S);

    patch_A = A.patch_A;
    patch_S = S.patch_S;

    rgb_A = cat(3, patch_A(:,:,23), patch_A(:,:,12), patch_A(:,:,5));
    rgb_S = cat(3, patch_S(:,:,4), patch_S(:,:,3), patch_S(:,:,2));

    % ===== 小檢查：快速看是不是全黑、scale 差太多、zero 太多 =====
    fprintf('\n================ Patch %05d ================\n', patch_id);
    quick_check('AVIRIS cube', patch_A);
    quick_check('S2 cube    ', patch_S);
    quick_check('AVIRIS RGB ', rgb_A);
    quick_check('S2 RGB     ', rgb_S);

    ratio_mean = mean(patch_A(:), 'omitnan') / (mean(patch_S(:), 'omitnan') + 1e-8);
    ratio_max  = max(patch_A(:), [], 'omitnan') / (max(patch_S(:), [], 'omitnan') + 1e-8);

    gray_A = mean(percentile_norm(rgb_A, 2, 98), 3);
    gray_S = mean(percentile_norm(rgb_S, 2, 98), 3);
    rgb_corr = corr(gray_A(:), gray_S(:), 'rows', 'complete');

    fprintf('mean(A)/mean(S) = %.4f | max(A)/max(S) = %.4f | RGB corr = %.4f\n', ...
        ratio_mean, ratio_max, rgb_corr);

    if use_shared_scale
        all_rgb = cat(4, rgb_A, rgb_S);
        mn = min(all_rgb(:), [], 'omitnan');
        mx = max(all_rgb(:), [], 'omitnan');

        rgb_A_show = clamp01((double(rgb_A) - mn) / (mx - mn + 1e-8));
        rgb_S_show = clamp01((double(rgb_S) - mn) / (mx - mn + 1e-8));
    else
        rgb_A_show = percentile_norm(rgb_A, 2, 98);
        rgb_S_show = percentile_norm(rgb_S, 2, 98);
    end

    figure('Name', sprintf('Patch %05d', patch_id), ...
           'Position', [100, 100, 1200, 500]);

    subplot(1,2,1);
    imshow(rgb_A_show);
    title(sprintf('AVIRIS | %s', file_name), 'Interpreter', 'none');

    subplot(1,2,2);
    imshow(rgb_S_show);
    title(sprintf('Sentinel-2 | %s', file_name), 'Interpreter', 'none');

    sgtitle(sprintf('Patch ID: %05d | shared scale = %d | RGB corr = %.3f', ...
        patch_id, use_shared_scale, rgb_corr));
end


% Local functions
function quick_check(name, x)
    x = double(x);
    fprintf('%s | min %.6f | max %.6f | mean %.6f | std %.6f | zero %.2f%% | NaN %.2f%%\n', ...
        name, ...
        min(x(:), [], 'omitnan'), ...
        max(x(:), [], 'omitnan'), ...
        mean(x(:), 'omitnan'), ...
        std(x(:), 'omitnan'), ...
        mean(abs(x(:)) < 1e-8, 'omitnan') * 100, ...
        mean(isnan(x(:))) * 100);
end

function img_out = percentile_norm(img, low_pct, high_pct)
    img = double(img);
    lo = prctile(img(:), low_pct);
    hi = prctile(img(:), high_pct);
    img_out = clamp01((img - lo) / (hi - lo + 1e-8));
end

function y = clamp01(x)
    y = min(max(x, 0), 1);
end
