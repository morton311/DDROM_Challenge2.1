%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Challenge 2.1: Time-Forecasting of a turbulent cavity flow
%
% Given the initial 70 snapshots of the planar velocity field
% from a turbulent cavity flow, forecast the subsequent 30 snapshots of the
% sequence.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluation script
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script performs two steps: 
% 
% (1) Validates the structure and basic integrity of a submitted HDF5 
%     results file (required datasets, dimensions, etc.).
% (2) If the test output file is available locally, computes the NMSE on 
%     the mean-removed (fluctuating) fields and generates summary plots.
%
% Note: Participants typically do NOT have access to the test output data,
% which is withheld for blind evaluation. In that case, this script will
% run the file-validation checks, print a message that the test output file
% is not available, and exit without computing error metrics.
%
% Datasets, challenge details and submission guidelines are
% maintained on the website:
% https://fluids-challenge.engin.umich.edu/
%
% 1/8/2026, initial version, OTS <oschmidt@ucsd.edu>

clear variables
clc

script_dir  = fileparts(mfilename('fullpath'));
data_dir    = fullfile(script_dir,'data');
method      = "LSTM"; % set to "DMD" or "LSTM"

switch method
    case "DMD"
        results_file = fullfile(script_dir,'DMD','Challenge2_1_test_DMD.h5');
    case "LSTM"
        results_file = fullfile(script_dir,'LSTM','Challenge2_1_test_LSTM.h5');
    otherwise
        error('Unsupported method "%s". Choose "DMD" or "LSTM".', method);
end

truth_file  = fullfile(data_dir,'Challenge2_1_test_output.h5');
grid_file   = fullfile(data_dir,'Challenge2_1_grid.h5');
param_file  = fullfile(data_dir,'Challenge2_1_parameters.h5');

% Validate forecast file structure
isValid = checkResultsFile(results_file);

if isValid
    fprintf('Reading results file... '); tic
    ux_est = h5read(results_file,'/ux');
    uy_est = h5read(results_file,'/uy');
    ux_mean = h5read(results_file,'/ux_mean');
    uy_mean = h5read(results_file,'/uy_mean');
    t_est  = h5read(results_file,'/t');
    fprintf([num2str(toc),' seconds \n']);
else
    return;
end


% Load ground-truth data (if provided)
isAvailable = ( exist(truth_file, 'file') == 2 );

if isAvailable
    fprintf('Reading test data... '); tic
    ux_true = h5read(truth_file,'/ux');
    uy_true = h5read(truth_file,'/uy');
    fprintf([num2str(toc),' seconds \n']);
else
    disp('The test output file is not available.');
    return;
end

% Basic compatibility checks
x  = h5read(grid_file,'/x');
y  = h5read(grid_file,'/y');

dt = h5read(param_file,'/dt');
[xx,yy] = meshgrid(x,y);

t_est = t_est(:).'; % ensure row vector
nt_fore_est = size(ux_est,3);
nt_fore_truth = size(ux_true,3);
n_seq_est = size(ux_est,4);
n_seq_truth = size(ux_true,4);

if nt_fore_est ~= nt_fore_truth || n_seq_est ~= n_seq_truth
    fprintf('Dimension mismatch: results file is [%d time steps x %d sequences], expected [%d x %d].\n', ...
        nt_fore_est, n_seq_est, nt_fore_truth, n_seq_truth);
    return;
end

if ~isequal(size(ux_est), size(uy_est))
    disp('Dimension mismatch between ux and uy forecasts.');
    return;
end

if ~isequal(size(ux_est), size(ux_true)) || ~isequal(size(uy_est), size(uy_true))
    disp('Results file dimensions do not match the ground truth data.');
    return;
end

expected_t = (1:nt_fore_truth) * dt;
if numel(t_est) ~= nt_fore_truth
    fprintf('Time vector length (%d) does not match forecast horizon (%d).\n', numel(t_est), nt_fore_truth);
elseif max(abs(t_est - expected_t)) > 1e-6
    fprintf('Warning: forecast time vector differs from expected values (max error = %.3e).\n', max(abs(t_est - expected_t)));
end

% Calculate NMSE per forecast step (using mean-removed fields)
[nx,ny,nt_fore,n_seq] = size(ux_true);
ux_true_mat = reshape(ux_true,[nx*ny, nt_fore, n_seq]);
uy_true_mat = reshape(uy_true,[nx*ny, nt_fore, n_seq]);
ux_est_mat  = reshape(ux_est, [nx*ny, nt_fore, n_seq]);
uy_est_mat  = reshape(uy_est, [nx*ny, nt_fore, n_seq]);

e_nmse = zeros(nt_fore,n_seq);
for i = 1:n_seq
    % Ground truth contains mean; subtract it to get fluctuations
    ux_true_this = ux_true_mat(:,:,i) - ux_mean(:);
    uy_true_this = uy_true_mat(:,:,i) - uy_mean(:);
    % Forecasts are already mean-removed (saved that way by LSTM/DMD scripts)
    ux_est_this  = ux_est_mat(:,:,i);
    uy_est_this  = uy_est_mat(:,:,i);

    q_true = [ux_true_this; uy_true_this];   % [2*nx*ny x nt_fore]
    q_est  = [ux_est_this;  uy_est_this ];

    diff_sq = (q_est - q_true).^2;
    true_sq = q_true.^2;
    e_nmse(:,i) = (mean(diff_sq,1) ./ (mean(true_sq,1) + eps)).';
end


% Figure: NMSE vs forecast step
rgb = lines;
f_error = figure;
set(f_error,'Name','Forecast NMSE');
hold on;
for i = 1:n_seq
    plot(e_nmse(:,i),'LineWidth',0.5,'Color',0.8*[1 1 1]);
end

e_nmse_mean = mean(e_nmse,2);
e_nmse_std  = std(e_nmse,0,2);
h_band = fill([1:nt_fore nt_fore:-1:1],[e_nmse_mean+e_nmse_std; flipud(e_nmse_mean-e_nmse_std)], ...
                rgb(1,:),'EdgeColor','none','FaceAlpha',0.1);
h_mean = plot(e_nmse_mean,'-','LineWidth',2,'Color',rgb(1,:));
box on;
xlabel('Forecast step','Interpreter','latex','FontSize',12);
ylabel('$e_\mathrm{NMSE}$','Interpreter','latex','FontSize',12);
title('Forecast NMSE across horizon','Interpreter','latex');
leg = legend([h_band h_mean], {'std','mean'}, 'Location','best');
set(leg,'Interpreter','latex');


% Figure: sample comparison
f_fields  = figure;
i_display = 1;
if i_display > n_seq
    error('Requested sequence %d exceeds available results (%d).', i_display, n_seq);
end
ti_plot   = [1 2 5 10 nt_fore];
ti_plot   = ti_plot(ti_plot <= nt_fore);
nt_plot   = numel(ti_plot);

ux_true_disp = ux_true(:,:,:,i_display);
uy_true_disp = uy_true(:,:,:,i_display);
ux_est_disp  = ux_est(:,:,:,i_display);
uy_est_disp  = uy_est(:,:,:,i_display);

tl_cmp = tiledlayout(4,nt_plot,'TileSpacing','compact');
title(tl_cmp, sprintf('Ground truth vs forecast for episode %d (%s)', i_display, method), ...
    'Interpreter','latex');

for k = 1:nt_plot
    ti = ti_plot(k);

    nexttile(k+0*nt_plot);
    pcolor(xx,yy,real(ux_true_disp(:,:,ti))); shading interp; axis equal tight;
    clim([-100 250]);
    set(gca,'YTick',[],'XTick',[]); box on;
    title(['$u^{(' num2str(ti) ')}(\mathbf{x})$'],'Interpreter','latex');

    nexttile(k+1*nt_plot);
    pcolor(xx,yy,real(ux_est_disp(:,:,ti)) + ux_mean); shading interp; axis equal tight;
    clim([-100 250]);
    set(gca,'YTick',[],'XTick',[]); box on;
    if method == "DMD"
        title(['$\tilde{u}_\mathrm{DMD}^{(' num2str(ti) ')}(\mathbf{x})$'],'Interpreter','latex');
    else
        title(['$\tilde{u}_\mathrm{LSTM}^{(' num2str(ti) ')}(\mathbf{x})$'],'Interpreter','latex');
    end

    nexttile(k+2*nt_plot);
    pcolor(xx,yy,real(uy_true_disp(:,:,ti))); shading interp; axis equal tight;
    clim(50*[-1 1]);
    set(gca,'YTick',[],'XTick',[]); box on;
    title(['$v^{(' num2str(ti) ')}(\mathbf{x})$'],'Interpreter','latex');

    nexttile(k+3*nt_plot);
    pcolor(xx,yy,real(uy_est_disp(:,:,ti))); shading interp; axis equal tight;
    clim(50*[-1 1]);
    set(gca,'YTick',[],'XTick',[]); box on;
    if method == "DMD"
        title(['$\tilde{v}_\mathrm{DMD}^{(' num2str(ti) ')}(\mathbf{x})$'],'Interpreter','latex');
    else
        title(['$\tilde{v}_\mathrm{LSTM}^{(' num2str(ti) ')}(\mathbf{x})$'],'Interpreter','latex');
    end
end
colormap parula;


% Validation helper
function isValid = checkResultsFile(filename)

if exist(filename, 'file') ~= 2
    fprintf('The specified HDF5 file does not exist: %s\n', filename);
    isValid = false;
    return;
end

isValid = true;

% Required datasets
requiredFields = {'/ux','/uy','/t'};

for i = 1:numel(requiredFields)
    field = requiredFields{i};
    try
        data = h5read(filename, field);
    catch
        fprintf('Field "%s" is missing from the results file.\n', field);
        isValid = false;
        continue;
    end

    if any(isnan(data(:)))
        fprintf('Field "%s" contains NaN values.\n', field);
        isValid = false;
    end
    if any(isinf(data(:)))
        fprintf('Field "%s" contains Inf values.\n', field);
        isValid = false;
    end

    switch field
        case '/ux'
            if ndims(data) ~= 4
                fprintf('Field "%s" must be a 4-D array [nx ny nt_fore n_seq].\n', field);
                isValid = false;
            end
            if ~isreal(data)
                fprintf('Field "%s" must contain real values.\n', field);
                isValid = false;
            end
        case '/uy'
            if ndims(data) ~= 4
                fprintf('Field "%s" must be a 4-D array [nx ny nt_fore n_seq].\n', field);
                isValid = false;
            end
            if ~isreal(data)
                fprintf('Field "%s" must contain real values.\n', field);
                isValid = false;
            end
        case '/t'
            if ~isvector(data)
                fprintf('Field "%s" must be a vector of forecast times.\n', field);
                isValid = false;
            end
    end
end

if isValid
    disp('Results file passed basic validation.');
else
    disp('Validation failed.');
end

end
