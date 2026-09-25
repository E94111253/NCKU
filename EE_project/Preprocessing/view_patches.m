clc; clear; close all;

%% ====== 路徑設定 ======
folder_A = 'crop_A';
folder_S = 'crop_S';

%% ====== 顯示模式 ======
% mode = 'A'     只看 AVIRIS
% mode = 'S'     只看 Sentinel-2
% mode = 'both'  同時看 A/S，一張 patch 顯示成左右對照
mode = 'S';

patches_per_page = 40;

files_A = dir(fullfile(folder_A, 'patch_*.mat'));
files_S = dir(fullfile(folder_S, 'patch_*.mat'));

if isempty(files_A)
    error('crop_A 裡面找不到 patch_*.mat');
end

if isempty(files_S)
    error('crop_S 裡面找不到 patch_*.mat');
end

%% 按檔名排序
[~, idxA] = sort({files_A.name});
[~, idxS] = sort({files_S.name});

files_A = files_A(idxA);
files_S = files_S(idxS);

if length(files_A) ~= length(files_S)
    warning('A / S 數量不一致：A=%d, S=%d', length(files_A), length(files_S));
end

N = min(length(files_A), length(files_S));
cache_A = cell(N, 1);
cache_S = cell(N, 1);
fprintf('剩下 patch 數量：%d\n', N);

%% ====== 分頁設定 ======
total_pages = ceil(N / patches_per_page);
page = 1;

while true

    start_idx = (page - 1) * patches_per_page + 1;
    end_idx = min(page * patches_per_page, N);

    num_show = end_idx - start_idx + 1;

    figure(1); clf;
    set(gcf, 'Position', [50, 100, 1600, 900]);

    %% 決定 subplot 排版
    n_col = 8;
    n_row = ceil(patches_per_page / n_col);

    for k = 1:num_show

        idx = start_idx + k - 1;

        %% ====== 取得實際 patch 編號 ======
        % 例如 patch_00023.mat -> patch_00023
        [~, patch_id, ~] = fileparts(files_S(idx).name);
        
        %% ====== 依照 mode 需要才讀資料，加快速度 ======
        if strcmpi(mode, 'A')
        
            if isempty(cache_A{idx})
                A = load(fullfile(folder_A, files_A(idx).name), 'patch_A');
                cache_A{idx} = make_aviris_rgb(A.patch_A);
            end
        
            show_img = cache_A{idx};
            title_text = sprintf('%s | A', patch_id);
        
        elseif strcmpi(mode, 'S')
        
            if isempty(cache_S{idx})
                S = load(fullfile(folder_S, files_S(idx).name), 'patch_S');
                cache_S{idx} = make_s2_rgb(S.patch_S);
            end
        
            show_img = cache_S{idx};
            title_text = sprintf('%s | S', patch_id);
        
        elseif strcmpi(mode, 'both')
        
            if isempty(cache_A{idx})
                A = load(fullfile(folder_A, files_A(idx).name), 'patch_A');
                cache_A{idx} = make_aviris_rgb(A.patch_A);
            end
        
            if isempty(cache_S{idx})
                S = load(fullfile(folder_S, files_S(idx).name), 'patch_S');
                cache_S{idx} = make_s2_rgb(S.patch_S);
            end
        
            show_img = [cache_A{idx}, cache_S{idx}];
            title_text = sprintf('%s | A/S', patch_id);
        
        else
            error('mode 只能是 A, S, 或 both');
        end

        %% ====== 顯示 ======
        subplot(n_row, n_col, k);
        imshow(show_img);
        title(title_text, 'FontSize', 9, 'Interpreter', 'none');

    end

    sgtitle(sprintf('Remaining patches | Page %d / %d | index %d - %d | n=next, p=previous, q=quit', ...
        page, total_pages, start_idx, end_idx), ...
        'FontSize', 14, 'FontWeight', 'bold');

    %% ====== 等待按鍵 ======
    waitforbuttonpress;
    key = get(gcf, 'CurrentCharacter');

    if key == 'n'
        if page < total_pages
            page = page + 1;
        else
            fprintf('已經是最後一頁。\n');
        end

    elseif key == 'p'
        if page > 1
            page = page - 1;
        else
            fprintf('已經是第一頁。\n');
        end

    elseif key == 'q'
        fprintf('結束瀏覽。\n');
        break;

    else
        fprintf('請按 n / p / q。\n');
    end
end

function rgb = make_aviris_rgb(patch_A)

    patch_A = single(patch_A);

    nBands = size(patch_A, 3);

    if nBands < 40
        error('patch_A band 數太少，無法建立 RGB。');
    end

    %% 顯示用 AVIRIS 近似 RGB
    % 這裡只是為了看地形，不影響資料本身
    % 如果你覺得水體太黑或顏色怪，可以改這三個 index
    r_idx = min(40, nBands);
    g_idx = min(25, nBands);
    b_idx = min(10, nBands);

    rgb = cat(3, ...
        patch_A(:,:,r_idx), ...
        patch_A(:,:,g_idx), ...
        patch_A(:,:,b_idx));

    rgb = robust_normalize_rgb(rgb);
end

function rgb = make_s2_rgb(patch_S)

    patch_S = single(patch_S);

    if size(patch_S, 3) < 4
        error('patch_S band 數不足，無法建立 Sentinel-2 RGB。');
    end

    %% Sentinel-2 true color
    % B4 = Red   = index 4
    % B3 = Green = index 3
    % B2 = Blue  = index 2
    rgb = cat(3, ...
        patch_S(:,:,4), ...
        patch_S(:,:,3), ...
        patch_S(:,:,2));

    rgb = robust_normalize_rgb(rgb);
end

function img = robust_normalize_rgb(img)

    img = single(img);
    img(~isfinite(img)) = 0;

    for ch = 1:size(img, 3)

        band = img(:,:,ch);

        low  = prctile(band(:), 1);
        high = prctile(band(:), 99);

        if abs(high - low) < 1e-8
            band = zeros(size(band), 'single');
        else
            band = (band - low) / (high - low + 1e-8);
        end

        band(band < 0) = 0;
        band(band > 1) = 1;

        img(:,:,ch) = band;
    end
end