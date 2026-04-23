%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Challenge 2.1: Time-Forecasting of a turbulent cavity flow
%
% Given the initial 70 snapshots of the planar velocity field
% from a turbulent cavity flow, forecast the subsequent 30 snapshots of the
% sequence.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ML method: Long short-term memory (LSTM) networks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script performs five steps:
%
% (1) Reads the training data and test input data
% (2) Obtains POD expansion coefficients for training and test data
% (3) Trains an LSTM network on the normalized POD coefficients
% (4) Performs closed-loop forecasting for all test episodes
% (5) Writes the result file, which serves as an example of the file
%     participants are expected to send to the challenge POC
%
% Datasets, challenge details and submission guidelines are
% maintained on the website:
% https://fluids-challenge.engin.umich.edu/
%
% 1/8/2026, initial version, OTS <oschmidt@ucsd.edu>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear variables
clc

script_dir  = fileparts(mfilename('fullpath'));
data_dir    = fullfile(script_dir,'..','data');

% Hindcast/forecast lengths
nt_hind     = 70;
nt_fore     = 30;
nt_episode  = nt_hind+nt_fore;

% Load data
ux_train        = h5read(fullfile(data_dir,"Challenge2_1_train.h5"),'/ux');
uy_train        = h5read(fullfile(data_dir,"Challenge2_1_train.h5"),'/uy');
ux_test_input   = h5read(fullfile(data_dir,"Challenge2_1_test_input.h5"),'/ux');
uy_test_input   = h5read(fullfile(data_dir,"Challenge2_1_test_input.h5"),'/uy');

x           = h5read(fullfile(data_dir,"Challenge2_1_grid.h5"),'/x');
y           = h5read(fullfile(data_dir,"Challenge2_1_grid.h5"),'/y');
dt          = h5read(fullfile(data_dir,'Challenge2_1_parameters.h5'),'/dt');

% Dimensions and 2D grid
[nx,ny,nt_train] = size(ux_train);
ndof        = 2*nx*ny;
[xx,yy]     = meshgrid(x,y);
n_sample_train  = floor(nt_train/nt_episode);
if n_sample_train < 1
    error('Training data does not contain a full hindcast/forecast block.');
end

function [net_out, A_fore] = forecastClosedLoop(net_in, A_norm, nt_fore)
% Helper: advance LSTM in closed loop for nt_fore steps
    net_out = resetState(net_in);
    X0 = A_norm(:,1:end-1).';           % [T x n_modes]
    X0_t = X0.';                        % [n_modes x T]
    [net_out, ~]        = predictAndUpdateState(net_out, X0_t);
    [net_out, y_prev]   = predictAndUpdateState(net_out, X0_t(:,end));

    n_modes = size(A_norm,1);
    A_fore = zeros(nt_fore,n_modes);
    A_fore(1,:) = y_prev(:).';
    for t = 2:nt_fore
        [net_out, y_t]  = predictAndUpdateState(net_out, A_fore(t-1,:).');
        A_fore(t,:) = y_t(:).';
    end
end

n_train_used    = n_sample_train * nt_episode;
n_sample_test   = size(ux_test_input,4);
ux_forecast = zeros(nx,ny,nt_fore,n_sample_test);
uy_forecast = zeros(nx,ny,nt_fore,n_sample_test);

% Model parameters
nt_test     = n_sample_test*nt_hind;
n_modes     = 25;
n_lstmLayer = 128;
n_lstmEpoch = 200;
n_miniBatch = 26; % ~7 iterations/epoch for 182 training samples

% Mean subtraction (train-set mean)
ux_mean     = mean(ux_train,3);
uy_mean     = mean(uy_train,3);
ux_train    = ux_train - ux_mean;
uy_train    = uy_train - uy_mean;
ux_test_input    = ux_test_input - ux_mean;
uy_test_input    = uy_test_input - uy_mean;

% Build data matrices
% Training data is sequential
Q_train     = [reshape(ux_train(:,:,1:nt_train),[nx*ny nt_train]); ...
               reshape(uy_train(:,:,1:nt_train),[nx*ny nt_train])];
% Testing data is segmented into independent realizations then flattened
Q_test_input = [reshape(ux_test_input,[nx*ny nt_test]); ...
                reshape(uy_test_input,[nx*ny nt_test])];

%% POD on training data
[Phi,Sigma,V] = svds(Q_train,n_modes,'largest');
A_train       = Sigma*V';              % time coefficients
lambda        = diag(Sigma).^2;        % singular value power

% Show spectrum of retained modes
figure
bar(1:n_modes, lambda, 'facecolor', 'b', 'EdgeColor','none'); hold on
set(gca,'YScale','log')
title('POD spectrum')
xlabel('mode'); ylabel('$\lambda$','Interpreter','latex')

%% Obtain POD expansion coefficients for test data
A_test      = Phi' * Q_test_input;

%% POD reconstruction demo for a single hindcast window
% Expectation management: LSTM can only reconstruct the flow field as well
% as the POD basis allows
i_display   = 1;
Q_hind_disp = [reshape(ux_test_input(:,:,:,i_display),[nx*ny nt_hind]); ...
               reshape(uy_test_input(:,:,:,i_display),[nx*ny nt_hind])];
A_hind_disp = Phi' * Q_hind_disp;
Q_rec_hind  = Phi * A_hind_disp;

t_i = ceil(nt_hind/2);

ux_rec      = reshape(Q_rec_hind(1:ndof/2,     t_i),nx,ny);
uy_rec      = reshape(Q_rec_hind(ndof/2+1:end, t_i),nx,ny);
ux_true     = reshape(Q_hind_disp(1:ndof/2,    t_i),nx,ny);
uy_true     = reshape(Q_hind_disp(ndof/2+1:end,t_i),nx,ny);

figure
tl_hind = tiledlayout(2,2,'TileSpacing','compact');
title(tl_hind, sprintf('POD reconstruction for episode %d, step %d', i_display, t_i), 'Interpreter','latex')
nexttile; pcolor(xx,yy,ux_true);  colorbar; shading interp; clim(50*[-1 1])
nexttile; pcolor(xx,yy,uy_true);  colorbar; shading interp; clim(50*[-1 1])
nexttile; pcolor(xx,yy,ux_rec);   colorbar; shading interp; clim(50*[-1 1])
nexttile; pcolor(xx,yy,uy_rec);   colorbar; shading interp; clim(50*[-1 1])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% POD-LSTM closed-loop forecasting %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Build hind windows for train/test
A_hind_train    = cell(n_sample_train,1);
for i=1:n_sample_train
    offset      = (i-1)*nt_episode; % sequential hind/forecast blocks
    t_idx       = (1:nt_hind) + offset;
    A_hind_train{i}   = A_train(:,t_idx);
end

A_hind_test    = cell(n_sample_test,1);
for i=1:n_sample_test
    offset      = (i-1)*nt_hind;
    t_idx       = (1:nt_hind) + offset;
    A_hind_test{i}  = A_test(:,t_idx);
end

% Normalize modal coefficients using training statistics
A_mu   = mean(A_train,2);
A_std  = std(A_train,0,2) + eps;

% Build one-step input/target sequences from hind segments (normalized)
A1 = cell(n_sample_train,1);
A2 = cell(n_sample_train,1);
for n = 1:n_sample_train
    A_this  = (A_hind_train{n} - A_mu) ./ A_std; % [n_modes x nt_hind]
    A1{n}   = A_this(:,1:end-1).';               % [T x n_modes]
    A2{n}   = A_this(:,2:end  ).';
end

% Define and train LSTM (using trainNetwork)
layers = [
    sequenceInputLayer(n_modes)
    lstmLayer(n_lstmLayer,'OutputMode','sequence')
    fullyConnectedLayer(n_modes)
    regressionLayer
];
options = trainingOptions('adam', ...
    'MaxEpochs', n_lstmEpoch, ...
    'Shuffle', 'every-epoch', ...
    'MiniBatchSize', n_miniBatch, ...
    'Verbose', false, ...
    'Plots', 'training-progress');
% Convert to features-first for trainNetwork
XTrain = cellfun(@transpose,A1,'UniformOutput',false); % [n_modes x T]
YTrain = cellfun(@transpose,A2,'UniformOutput',false);
net = trainNetwork(XTrain,YTrain,layers,options);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Training forecast error analysis       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f_error_train = figure;
e_nmse_train  = zeros(nt_fore,n_sample_train);
A_fore_train  = zeros(n_sample_train,nt_fore,n_modes);

for n = 1:n_sample_train
    idx_start = (n-1)*nt_episode + 1;
    idx_fore  = (idx_start+nt_hind):(idx_start+nt_hind+nt_fore-1);

    Q_fore_true = [reshape(ux_train(:,:,idx_fore),[nx*ny nt_fore]); ...
                   reshape(uy_train(:,:,idx_fore),[nx*ny nt_fore])];

    A_in = (A_hind_train{n} - A_mu) ./ A_std; % [n_modes x nt_hind]
    [~, A_fore] = forecastClosedLoop(net, A_in, nt_fore);
    A_fore = A_fore .* A_std.' + A_mu.'; % [nt_fore x n_modes]
    A_fore_train(n,:,:) = A_fore;

    Q_rec = Phi * A_fore.'; % [ndof x nt_fore]
    for ti = 1:nt_fore
        q_true = Q_fore_true(:,ti);
        q_hat  = Q_rec(:,ti);
        denom = mean(q_true.^2) + eps;           % normalize by true energy
        e_nmse_train(ti,n) = mean((q_true - q_hat).^2) / denom;
    end

    figure(f_error_train)
    set(f_error_train,'Name',sprintf('%d/%d (training)',n,n_sample_train))
    plot(e_nmse_train(:,n),'LineWidth',0.5,'Color',0.8*[1 1 1]); hold on
    drawnow
end

figure(f_error_train);
hold on;

rgb = lines;
e_nmse_mean = mean(e_nmse_train,2);
e_nmse_std  = std(e_nmse_train,0,2);
h_band = fill([1:nt_fore nt_fore:-1:1],[e_nmse_mean+e_nmse_std; flipud(e_nmse_mean-e_nmse_std)], ...
             rgb(2,:),'EdgeColor','none','FaceAlpha',0.1);
h_mean = plot(e_nmse_mean,'r-','LineWidth',2,'Color',rgb(2,:));
xlabel('$\Delta t$','Interpreter','latex'); ylabel('$e_\mathrm{NMSE}$','Interpreter','latex')
title('LSTM forecast NMSE across forecast horizon (training data)','Interpreter','latex')
if ~isempty(h_band) && ~isempty(h_mean)
    leg = legend([h_band h_mean], 'std','mean','Location','best');
    set(leg,'Interpreter','latex');
end
train_nmse_mean = mean(e_nmse_train(:));
train_nmse_std  = std(e_nmse_train(:));
fprintf('Training NMSE (mean +/- std over all blocks/horizons): %.3e +/- %.3e\n', ...
    train_nmse_mean, train_nmse_std);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% POD-LSTM closed-loop forecasting %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Forecast test episodes (no ground truth available)
A_fore_test = zeros(n_sample_test,nt_fore,n_modes);
for n = 1:n_sample_test
    A_in = (A_hind_test{n} - A_mu) ./ A_std; % [n_modes x nt_hind]
    [~, A_fore] = forecastClosedLoop(net, A_in, nt_fore);
    A_fore = A_fore .* A_std.' + A_mu.'; % [nt_fore x n_modes]
    A_fore_test(n,:,:) = A_fore;

    Q_rec = Phi * A_fore.'; % [ndof x nt_fore]
    for ti = 1:nt_fore
        q_hat = Q_rec(:,ti);
        ux_forecast(:,:,ti,n) = reshape(q_hat(1:nx*ny),[nx ny]);
        uy_forecast(:,:,ti,n) = reshape(q_hat(nx*ny+1:end),[nx ny]);
    end
end

% Quick sanity plot of coefficients for first test episode
figure
plot(1:nt_hind, A_hind_test{1}.'), hold on
plot(nt_hind+1:nt_hind+nt_fore, squeeze(A_fore_test(1,:,:)))
title('Hind- and forecast POD coefficients for episode 1','Interpreter','latex')
xlabel('time step'); ylabel('coefficient value')

%% Export forecast to HDF5 for evaluation
t_forecast = (1:nt_fore) * dt;
results_file = fullfile(script_dir,'Challenge2_1_test_LSTM.h5');
if exist(results_file,'file') == 2
    delete(results_file);
end
h5create(results_file,'/ux',[nx ny nt_fore n_sample_test],'Datatype','double');
h5write(results_file,'/ux',ux_forecast);
h5create(results_file,'/uy',[nx ny nt_fore n_sample_test],'Datatype','double');
h5write(results_file,'/uy',uy_forecast);
h5create(results_file,'/ux_mean',[nx ny],'Datatype','double');
h5write(results_file,'/ux_mean',ux_mean);
h5create(results_file,'/uy_mean',[nx ny],'Datatype','double');
h5write(results_file,'/uy_mean',uy_mean);
h5create(results_file,'/t',[1 nt_fore],'Datatype','double');
h5write(results_file,'/t',t_forecast);
h5create(results_file,'/x',size(x),'Datatype','double');
h5write(results_file,'/x',x);
h5create(results_file,'/y',size(y),'Datatype','double');
h5write(results_file,'/y',y);
h5writeatt(results_file,'/','method','POD-LSTM forecast');
h5writeatt(results_file,'/','nt_hind',nt_hind);
h5writeatt(results_file,'/','nt_fore',nt_fore);
fprintf('Saved LSTM forecast to %s\n', results_file);

%% Compare LSTM prediction to true flow field (training block)
% Use the same display sample as above (and as DMD)
i_display = 1;
if i_display > n_sample_train
    error('Requested training block %d exceeds available blocks (%d).', i_display, n_sample_train);
end

figure
ti_plot = [1 2 5 10 nt_fore];
ti_plot = ti_plot(ti_plot <= nt_fore);
nt_plot  = numel(ti_plot);

tl_cmp = tiledlayout(4,nt_plot,'TileSpacing','compact');
title(tl_cmp, sprintf('Training block %d: ground truth vs LSTM', i_display), 'Interpreter','latex')

idx_start_d = (i_display-1)*nt_episode + 1;
idx_fore_d  = (idx_start_d+nt_hind):(idx_start_d+nt_hind+nt_fore-1);

A_fore_train_disp = squeeze(A_fore_train(i_display,:,:)); % [nt_fore x n_modes]
Q_rec_disp        = Phi * A_fore_train_disp.';            % [ndof x nt_fore]
Q_true_disp       = [reshape(ux_train(:,:,idx_fore_d),[nx*ny nt_fore]); ...
                     reshape(uy_train(:,:,idx_fore_d),[nx*ny nt_fore])];

for i = 1:nt_plot
    ti     = ti_plot(i);
    q_LSTM = Q_rec_disp(:,ti);
    q_true = Q_true_disp(:,ti);

    nexttile(i+0*nt_plot)
    u_x_this = reshape(q_true(1:nx*ny),[nx ny]) + ux_mean;
    pcolor(xx,yy,real(u_x_this)); shading interp; axis equal tight
    clim([-100 250])
    set(gca,'YTick',[],'XTick',[]); box on
    title(['$u^{(' num2str(ti) ')}(\mathbf{x})$'],'Interpreter','latex')

    % LSTM u_x
    nexttile(i+1*nt_plot)
    u_x_predict = reshape(q_LSTM(1:nx*ny),[nx ny]) + ux_mean;
    pcolor(xx,yy,real(u_x_predict)); shading interp; axis equal tight
    clim([-100 250])
    set(gca,'YTick',[],'XTick',[]); box on
    title(['$\tilde{u}_{\mathrm{LSTM}}^{(' num2str(ti) ')}(\mathbf{x})$'],'Interpreter','latex')

    % True u_y
    nexttile(i+2*nt_plot)
    u_y_this = reshape(q_true(nx*ny+1:end),[nx ny]) + uy_mean;
    pcolor(xx,yy,real(u_y_this)); shading interp; axis equal tight
    clim(50*[-1 1])
    set(gca,'YTick',[],'XTick',[]); box on
    title(['$v^{(' num2str(ti) ')}(\mathbf{x})$'],'Interpreter','latex')

    % LSTM u_y
    nexttile(i+3*nt_plot)
    u_y_predict = reshape(q_LSTM(nx*ny+1:end),[nx ny]) + uy_mean;
    pcolor(xx,yy,real(u_y_predict)); shading interp; axis equal tight
    clim(50*[-1 1])
    set(gca,'YTick',[],'XTick',[]); box on
    title(['$\tilde{v}_{\mathrm{LSTM}}^{(' num2str(ti) ')}(\mathbf{x})$'],'Interpreter','latex')

    colormap parula
end
