clc; clear;

%% ============================================================
% crop_clean_real_mode.m
%
% mode = "simulation": AVIRIS -> simulated Sentinel-2
% mode = "real"      : aligned AVIRIS + aligned real Sentinel-2
%
%% ============================================================

%% ===================== User settings ========================
mode = "real";              % "simulation" or "real"

aviris_folder = fullfile('Dataset', 'aviris');
s2_folder     = fullfile('Dataset', 's2');
pair_map_file = 's2_pair_map.csv';

if mode == "real"
    output_A = 'crop_A_r';
    output_S = 'crop_S_r';
else
    output_A = 'crop_A';
    output_S = 'crop_S';
end

patch_size = 256;
stride = 256;

% AVIRIS original 224 bands -> remove corrupted / absorption bands -> 172 bands
remove_bands = unique([1:10, 104:116, 152:170, 215:224]);
keep_bands = setdiff(1:224, remove_bands);


% use [1:10,12:13] to exclude B10.
s2_band_idx = 1:12;
% s2_band_idx = [1:10, 12:13];

% S2 spectral settings for simulation mode only
S2_center_nm = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190];
S2_bw_nm     = [20,  65,  35,  30,  15,  15,  20, 115, 20,  20,  90,   180];
idx_20m = [5, 6, 7, 9, 11, 12];
idx_60m = [1, 10];


if ~exist(output_A, 'dir'), mkdir(output_A); end
if ~exist(output_S, 'dir'), mkdir(output_S); end

if numel(keep_bands) ~= 172
    error('keep_bands should contain 172 bands, but got %d.', numel(keep_bands));
end

%% ===================== File list ============================
if mode == "real"
    opts = detectImportOptions(pair_map_file, 'FileType', 'text', 'Delimiter', ',');
    opts.VariableNamesLine = 1;
    opts.DataLines = [2 Inf];
    opts = setvartype(opts, 'string');

    T = readtable(pair_map_file, opts);

    disp(T.Properties.VariableNames)
    disp(T(1:min(3,height(T)),:))
    n_files = height(T);
else
    files = dir(fullfile(aviris_folder, '*.dat'));
    n_files = numel(files);
end

patch_count = 0;

%% ===================== Main loop ============================
for f = 1:n_files

    if mode == "real"
        aviris_name = char(T.aviris_file(f));
        s2_name     = char(T.s2_file(f));
    else
        aviris_name = files(f).name;
        s2_name     = '';
    end

    [~, aviris_base, ~] = fileparts(aviris_name);

    aviris_dat = fullfile(aviris_folder, aviris_name);
    aviris_hdr = fullfile(aviris_folder, [aviris_base, '.hdr']);

    fprintf('\nProcessing AVIRIS: %s\n', aviris_name);

    info_A = read_envi_hdr(aviris_hdr);
    machine_A = get_machine_format(info_A);
    fid_A = fopen(aviris_dat, 'r', machine_A);

    if fid_A < 0
        error('Cannot open AVIRIS file: %s', aviris_dat);
    end

    H_A = info_A.lines;
    W_A = info_A.samples;
    B_A = info_A.bands;

    if B_A < 224
        fclose(fid_A);
        error('AVIRIS file has fewer than 224 bands: %s', aviris_name);
    end

    %% AVIRIS wavelength for simulation mapping
    if isfield(info_A, 'wavelength') && numel(info_A.wavelength) >= 224
        aviris_wl_224 = info_A.wavelength(1:224);
        if max(aviris_wl_224) < 20
            aviris_wl_224 = aviris_wl_224 * 1000;
        end
    else
        aviris_wl_224 = linspace(400, 2500, 224);
    end
    aviris_wl_172 = aviris_wl_224(keep_bands);
    S2_map = build_s2_subset_map(aviris_wl_172, S2_center_nm, S2_bw_nm);

    %% Real S2 file
    if mode == "real"
        [~, s2_base, ~] = fileparts(s2_name);
        s2_dat = fullfile(s2_folder, s2_name);
        s2_hdr = fullfile(s2_folder, [s2_base, '.hdr']);

        fprintf('Using aligned S2: %s\n', s2_name);

        info_S = read_envi_hdr(s2_hdr);
        machine_S = get_machine_format(info_S);
        fid_S = fopen(s2_dat, 'r', machine_S);

        if fid_S < 0
            fclose(fid_A);
            error('Cannot open S2 file: %s', s2_dat);
        end

        H_use = min(H_A, info_S.lines);
        W_use = min(W_A, info_S.samples);
    else
        fid_S = [];
        info_S = [];
        H_use = H_A;
        W_use = W_A;
    end
    fprintf('AVIRIS: H=%d W=%d B=%d interleave=%s\n', ...
    info_A.lines, info_A.samples, info_A.bands, string(info_A.interleave));

    if mode == "real"
        fprintf('S2    : H=%d W=%d B=%d interleave=%s\n', ...
            info_S.lines, info_S.samples, info_S.bands, string(info_S.interleave));
    else
        fprintf('S2    : simulated from AVIRIS\n');
    end
    
    fprintf('H_use=%d W_use=%d\n', H_use, W_use);
    %% Crop patches
    for r = 1:stride:(H_use - patch_size + 1)
        for c = 1:stride:(W_use - patch_size + 1)

            patch_A = read_envi_patch(fid_A, info_A, r, c, patch_size, keep_bands);
            if max(patch_A(:), [], 'omitnan') > 1.5
                patch_A = patch_A / 10000;
            end

            if mode == "real"
                patch_S = read_envi_patch(fid_S, info_S, r, c, patch_size, s2_band_idx);

    
                if max(patch_S(:), [], 'omitnan') > 1.5
                    patch_S = patch_S / 10000;
                end

                data_mode = 'real_aligned_s2';
                s2_source_file = s2_name;
            else
                patch_S = simulate_s2_from_aviris(patch_A, S2_map);
                [h, w, ~] = size(patch_S);

                for b = idx_20m
                    patch_S(:,:,b) = uniform_degrade_and_copy(patch_S(:,:,b), 2, [h, w]);
                end

                for b = idx_60m
                    patch_S(:,:,b) = uniform_degrade_and_copy(patch_S(:,:,b), 6, [h, w]);
                end

                data_mode = 'simulation_aviris_to_s2';
                s2_source_file = 'simulated_from_aviris';
            end
            %% ====== Basic bad-patch filter: only remove obvious NoData / black border ======
            A_vals = patch_A(:);
            S_vals = patch_S(:);
            
            if any(~isfinite(A_vals)) || any(~isfinite(S_vals))
                fprintf('Skip patch at r=%d c=%d: NaN/Inf\n', r, c);
                continue;
            end
            
            A_max  = max(A_vals, [], 'omitnan');
            S_max  = max(S_vals, [], 'omitnan');
            A_std  = std(double(A_vals), 'omitnan');
            S_std  = std(double(S_vals), 'omitnan');
            A_zero = nnz(abs(A_vals) <= 1e-8) / numel(A_vals);
            S_zero = nnz(abs(S_vals) <= 1e-8) / numel(S_vals);
            
            % Do NOT remove normal dark water.
            % Only remove patches that are nearly all zero or almost no variation.
            if A_zero > 0.50 || S_zero > 0.50
                fprintf('Skip patch at r=%d c=%d: too many zeros A=%.2f S=%.2f\n', ...
                    r, c, A_zero, S_zero);
                continue;
            end
            
            if A_max < 1e-5 && A_std < 1e-6
                fprintf('Skip patch at r=%d c=%d: AVIRIS almost black\n', r, c);
                continue;
            end
            
            if S_max < 1e-5 && S_std < 1e-6
                fprintf('Skip patch at r=%d c=%d: S2 almost black\n', r, c);
                continue;
            end

            patch_count = patch_count + 1;
            fname = sprintf('patch_%05d.mat', patch_count);

            source_file = aviris_name;
            source_base = aviris_base;
            patch_index = patch_count;

            meta = struct();
            meta.mode = data_mode;
            meta.source_file = aviris_name;
            meta.source_base = aviris_base;
            meta.s2_source_file = s2_source_file;
            meta.patch_size = patch_size;
            meta.stride = stride;
            meta.patch_index = patch_count;
            meta.keep_bands = keep_bands;
            meta.remove_bands = remove_bands;
            meta.s2_band_idx = s2_band_idx;

            save(fullfile(output_A, fname), ...
                'patch_A', ...
                'source_file', ...
                'source_base', ...
                'patch_index', ...
                'keep_bands', ...
                'remove_bands', ...
                'meta', ...
                '-v7.3');
            
            save(fullfile(output_S, fname), ...
                'patch_S', ...
                'source_file', ...
                'source_base', ...
                'patch_index', ...
                's2_source_file', ...
                's2_band_idx', ...
                'meta', ...
                '-v7.3');
            
            fprintf('Saved %s\n', fname);
        end
    end

    fclose(fid_A);
    if mode == "real"
        fclose(fid_S);
    end
end

fprintf('\nDone. Total patches: %d\n', patch_count);

%% ============================================================
% Local functions
%% ============================================================

function info = read_envi_hdr(hdr_file)
    txt = fileread(hdr_file);
    lines = regexp(txt, '\r\n|\n|\r', 'split');

    info = struct();
    collecting_key = '';
    collecting_value = '';

    for i = 1:length(lines)
        line = strtrim(lines{i});
        if isempty(line) || startsWith(line, ';')
            continue;
        end

        if ~isempty(collecting_key)
            collecting_value = [collecting_value, ' ', line]; %#ok<AGROW>
            if contains(line, '}')
                info.(collecting_key) = parse_hdr_value(collecting_value);
                collecting_key = '';
                collecting_value = '';
            end
            continue;
        end

        eq_pos = strfind(line, '=');
        if isempty(eq_pos)
            continue;
        end

        key = strtrim(line(1:eq_pos(1)-1));
        val = strtrim(line(eq_pos(1)+1:end));
        key = lower(strrep(key, ' ', '_'));

        if contains(val, '{') && ~contains(val, '}')
            collecting_key = key;
            collecting_value = val;
        else
            info.(key) = parse_hdr_value(val);
        end
    end

    if ~isfield(info, 'header_offset')
        info.header_offset = 0;
    end
end

function value = parse_hdr_value(val)
    val = strtrim(val);

    if startsWith(val, '{') && endsWith(val, '}')
        val = erase(val, {'{', '}'});
        parts = regexp(val, ',', 'split');
        parts = strtrim(parts);
        nums = str2double(parts);

        if all(~isnan(nums))
            value = nums;
        else
            value = parts;
        end
        return;
    end

    num = str2double(val);
    if ~isnan(num)
        value = num;
    else
        value = val;
    end
end

function machine = get_machine_format(info)
    if info.byte_order == 0
        machine = 'ieee-le';
    else
        machine = 'ieee-be';
    end
end

function [source_prec, bytes] = envi_precision(data_type)
    switch data_type
        case 1
            source_prec = 'uint8';  bytes = 1;
        case 2
            source_prec = 'int16';  bytes = 2;
        case 3
            source_prec = 'int32';  bytes = 4;
        case 4
            source_prec = 'single'; bytes = 4;
        case 5
            source_prec = 'double'; bytes = 8;
        case 12
            source_prec = 'uint16'; bytes = 2;
        case 13
            source_prec = 'uint32'; bytes = 4;
        otherwise
            error('Unsupported ENVI data_type = %d', data_type);
    end
end

function patch = read_envi_patch(fid, info, row0, col0, patch_size, band_idx)
    [source_prec, bytes] = envi_precision(info.data_type);
    precision = sprintf('%s=>single', source_prec);

    lines = info.lines;
    samples = info.samples;
    bands = info.bands;
    header_offset = info.header_offset;
    interleave = lower(strtrim(info.interleave));

    nb = numel(band_idx);
    patch = zeros(patch_size, patch_size, nb, 'single');
    col_idx = col0:(col0 + patch_size - 1);

    switch interleave
        case 'bil'
            for rr = 1:patch_size
                line_idx = row0 + rr - 1;
                offset_values = (line_idx - 1) * bands * samples;
                fseek(fid, header_offset + offset_values * bytes, 'bof');
                line_raw = fread(fid, [samples, bands], precision);
                line_patch = line_raw(col_idx, band_idx);
                patch(rr,:,:) = reshape(line_patch, [1, patch_size, nb]);
            end

        case 'bip'
            for rr = 1:patch_size
                line_idx = row0 + rr - 1;
                offset_values = (line_idx - 1) * samples * bands;
                fseek(fid, header_offset + offset_values * bytes, 'bof');
                line_raw = fread(fid, [bands, samples], precision)';
                line_patch = line_raw(col_idx, band_idx);
                patch(rr,:,:) = reshape(line_patch, [1, patch_size, nb]);
            end

        case 'bsq'
            for bb = 1:nb
                b = band_idx(bb);
                for rr = 1:patch_size
                    line_idx = row0 + rr - 1;
                    offset_values = (b - 1) * lines * samples + ...
                                    (line_idx - 1) * samples + ...
                                    (col0 - 1);
                    fseek(fid, header_offset + offset_values * bytes, 'bof');
                    vals = fread(fid, [1, patch_size], precision);
                    patch(rr,:,bb) = vals;
                end
            end

        otherwise
            error('Unsupported interleave: %s', interleave);
    end
end

function S2_map = build_s2_subset_map(aviris_wl_172, S2_center_nm, S2_bw_nm)
    nS = numel(S2_center_nm);
    nH = numel(aviris_wl_172);
    S2_map = zeros(nS, nH, 'single');

    for i = 1:nS
        low = S2_center_nm(i) - S2_bw_nm(i) / 2;
        high = S2_center_nm(i) + S2_bw_nm(i) / 2;
        idx = find(aviris_wl_172 >= low & aviris_wl_172 <= high);

        if isempty(idx)
            [~, idx] = min(abs(aviris_wl_172 - S2_center_nm(i)));
        end

        S2_map(i, idx) = 1 / numel(idx);
    end
end

function patch_S = simulate_s2_from_aviris(patch_A, S2_map)
    [h, w, ~] = size(patch_A);
    X = reshape(patch_A, [], 172)';
    Y = S2_map * X;
    patch_S = reshape(Y', h, w, 12);
    patch_S = single(patch_S);
end

function out = uniform_degrade_and_copy(band, factor, target_size)
    band = single(band);
    [h, w] = size(band);
    h2 = floor(h / factor) * factor;
    w2 = floor(w / factor) * factor;
    band_crop = band(1:h2, 1:w2);

    temp = reshape(band_crop, factor, h2/factor, factor, w2/factor);
    low = squeeze(mean(mean(temp, 1, 'omitnan'), 3, 'omitnan'));
    up = kron(low, ones(factor, factor, 'single'));

    out = zeros(target_size, 'single');
    out(1:size(up,1), 1:size(up,2)) = up;

    if size(up,1) < target_size(1) || size(up,2) < target_size(2)
        out = imresize(up, target_size, 'nearest');
    end
end
