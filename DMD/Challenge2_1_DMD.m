%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Challenge 2.1: Time-Forecasting of a turbulent cavity flow
%
% Given the initial 70 snapshots of the planar velocity field
% from a turbulent cavity flow, forecast the subsequent 30 snapshots of the
% sequence.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Classical method: Dynamic Mode Decomposition (DMD)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script performs four steps:
%
% (1) Reads the training data and test input data
% (2) Computes DMD modes from each hindcast window
% (3) Forecasts using the DMD eigenvalues and modes
% (4) Writes the result file, which serves as an example of the file
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

script_dir      = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir,'utils'));
data_dir        = fullfile(script_dir,'..','data');

ux_test_input   = h5read(fullfile(data_dir,"Challenge2_1_test_input.h5"),'/ux');
uy_test_input   = h5read(fullfile(data_dir,"Challenge2_1_test_input.h5"),'/uy');

ux_train        = h5read(fullfile(data_dir,"Challenge2_1_train.h5"),'/ux');
uy_train        = h5read(fullfile(data_dir,"Challenge2_1_train.h5"),'/uy');
%%

x               = h5read(fullfile(data_dir,"Challenge2_1_grid.h5"),'/x');
y               = h5read(fullfile(data_dir,"Challenge2_1_grid.h5"),'/y');
dt              = h5read(fullfile(data_dir,'Challenge2_1_parameters.h5'),'/dt');

% Mean subtraction (train-set mean)
ux_mean     = mean(ux_train,3);
uy_mean     = mean(uy_train,3);
ux_train    = ux_train - ux_mean;
uy_train    = uy_train - uy_mean;
ux_test_input    = ux_test_input - ux_mean;
uy_test_input    = uy_test_input - uy_mean;

% dimensions and 2D grid
[nx,ny,~,~] = size(ux_test_input);
[xx,yy]     = meshgrid(x,y);

f_error     = figure;

% Parameters
nt_hind     = 70;
nt_fore     = 30;
n_modes     = 25;

n_test_sample = size(ux_test_input,4);

% Use the training data to quantify DMD error
nt_block    = nt_hind + nt_fore; % alternate hind/forecast segments sequentially
nt_train    = size(ux_train,3);
n_train_seq = floor(nt_train / nt_block);
n_train_used= n_train_seq * nt_block;
e_nmse_train = zeros(nt_fore,n_train_seq);

for i_train = 1:n_train_seq

    % build data matrices from sequential training blocks
    idx_start = (i_train-1)*nt_block + 1;
    idx_hind  = idx_start:(idx_start+nt_hind-1);
    idx_fore  = (idx_start+nt_hind):(idx_start+nt_hind+nt_fore-1);

    Q_hind = [reshape(ux_train(:,:,idx_hind),[nx*ny nt_hind]); ...
              reshape(uy_train(:,:,idx_hind),[nx*ny nt_hind])];
    Q_fore = [reshape(ux_train(:,:,idx_fore),[nx*ny nt_fore]); ...
              reshape(uy_train(:,:,idx_fore),[nx*ny nt_fore])];

    %%%%%%%%
    %% DMD %
    %%%%%%%%
    [Psi,~,L,~] = dmd(Q_hind,dt,n_modes);

    % Use last hindcast snapshot as IC to avoid future leakage
    q_IC   = Q_hind(:,end);
    b      = Psi\q_IC;
    lambda = L; % discrete-time eigenvalues

    for ti = 1:nt_fore
        q_true = Q_fore(:,ti);
        % Predict ti steps ahead using discrete eigenvalues
        q_DMD  = real(Psi * (b .* (lambda.^ti))); % avoids repeated exp() calls
        denom = mean(q_true.^2) + eps;            % normalize by true energy
        e_nmse_train(ti,i_train) = mean((q_true - q_DMD).^2) / denom;
    end

    %% plot error evolution for this training block
    figure(f_error)
    set(f_error,'Name',sprintf('%d/%d (training)',i_train,n_train_seq))
    plot(e_nmse_train(:,i_train),'LineWidth',0.5,'Color',0.8*[1 1 1]), hold on
    drawnow
end

%% add mean and standard deviation of error to plot
e_nmse_mean = mean(e_nmse_train,2);
e_nmse_std  = std(e_nmse_train,0,2);
figure(f_error);
hold on;
rgb = lines;
h_band = fill([1:nt_fore nt_fore:-1:1],[e_nmse_mean+e_nmse_std; flipud(e_nmse_mean-e_nmse_std)], ...
                rgb(1,:),'EdgeColor','none','FaceAlpha',0.1);
h_mean = plot(e_nmse_mean,'-','LineWidth',2,'Color',rgb(1,:));
title('DMD forecast NMSE across forecast horizon (training data)','Interpreter','latex')
xlabel('$\Delta t$','Interpreter','latex'), ylabel('$e_\mathrm{NMSE}$','Interpreter','latex')
% Add legend for mean and standard deviation band
if ~isempty(h_band) && ~isempty(h_mean)
    leg = legend([h_band h_mean], 'std','mean','Location','best');
    set(leg,'Interpreter','latex');
end

train_nmse_mean = mean(e_nmse_train(:));
train_nmse_std  = std(e_nmse_train(:));
fprintf('Training NMSE (mean +/- std over all blocks/horizons): %.3e +/- %.3e\n', ...
    train_nmse_mean, train_nmse_std);

%% Forecast the test inputs (no ground truth available)
ux_forecast = zeros(nx,ny,nt_fore,n_test_sample);
uy_forecast = zeros(nx,ny,nt_fore,n_test_sample);

for i_test = 1:n_test_sample

    Q_hind = [reshape(ux_test_input(:,:,:,i_test),[nx*ny nt_hind]); ...
              reshape(uy_test_input(:,:,:,i_test),[nx*ny nt_hind])];

    [Psi,~,L,~] = dmd(Q_hind,dt,n_modes);

    q_IC   = Q_hind(:,end);
    b      = Psi\q_IC;
    lambda = L;

    for ti = 1:nt_fore
        q_DMD = real(Psi * (b .* (lambda.^ti))); % avoids repeated exp() calls
        ux_forecast(:,:,ti,i_test) = reshape(q_DMD(1:nx*ny),[nx ny]);
        uy_forecast(:,:,ti,i_test) = reshape(q_DMD(nx*ny+1:end),[nx ny]);
    end
end

%% Export forecast to HDF5 for evaluation
t_forecast = (1:nt_fore) * dt;
results_file = fullfile(script_dir,'Challenge2_1_test_DMD.h5');
if exist(results_file,'file') == 2
    delete(results_file);
end
h5create(results_file,'/ux',[nx ny nt_fore n_test_sample],'Datatype','double');
h5write(results_file,'/ux',ux_forecast);
h5create(results_file,'/uy',[nx ny nt_fore n_test_sample],'Datatype','double');
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
h5writeatt(results_file,'/','method','DMD baseline forecast');
h5writeatt(results_file,'/','nt_hind',nt_hind);
h5writeatt(results_file,'/','nt_fore',nt_fore);
fprintf('Saved DMD forecast to %s\n', results_file);

%% Compare DMD prediction to true flow field (training block)
figure
% Choose which training block to visualize explicitly
i_display = 1;
if i_display > n_train_seq
    error('Requested training block %d exceeds available blocks (%d).', i_display, n_train_seq);
end
idx_start_d = (i_display-1)*nt_block + 1;
idx_hind_d  = idx_start_d:(idx_start_d+nt_hind-1);
idx_fore_d  = (idx_start_d+nt_hind):(idx_start_d+nt_hind+nt_fore-1);

Q_hind_d = [reshape(ux_train(:,:,idx_hind_d),[nx*ny nt_hind]); ...
            reshape(uy_train(:,:,idx_hind_d),[nx*ny nt_hind])];
Q_fore_d = [reshape(ux_train(:,:,idx_fore_d),[nx*ny nt_fore]); ...
            reshape(uy_train(:,:,idx_fore_d),[nx*ny nt_fore])];
[Psi_d,~,L_d,~] = dmd(Q_hind_d,dt,n_modes);
qIC_d    = Q_hind_d(:,end);
b_d      = Psi_d\qIC_d;
lambda_d = L_d;

ti_plot = [1 2 5 10 nt_fore];
ti_plot = ti_plot(ti_plot <= nt_fore);
nt_plot  = numel(ti_plot);

tl_cmp = tiledlayout(4,nt_plot,'TileSpacing','compact');
title(tl_cmp, sprintf('Training block %d: ground truth vs DMD', i_display), 'Interpreter','latex')

for k = 1:nt_plot
    ti    = ti_plot(k);
    q_DMD = real(Psi_d * (b_d .* (lambda_d.^ti)));

    nexttile(k+0*nt_plot)
    u_x_this = reshape(Q_fore_d(1:nx*ny,ti),[nx ny])  + ux_mean;
    pcolor(xx,yy,real(u_x_this)); shading interp, axis equal tight
    clim([-100 250])
    set(gca,'YTick',[],'XTick',[]); box on
    title(['$u^{(' num2str(ti) ')}(\mathbf{x})$'],'Interpreter','latex')

    nexttile(k+2*nt_plot)
    u_y_this = reshape(Q_fore_d(nx*ny+1:end,ti),[nx ny]) + uy_mean;
    pcolor(xx,yy,real(u_y_this)); shading interp, axis equal tight
    clim(50*[-1 1])
    set(gca,'YTick',[],'XTick',[]); box on
    title(['$v^{(' num2str(ti) ')}(\mathbf{x})$'],'Interpreter','latex')

    nexttile(k+1*nt_plot)
    u_x_DMD = reshape(q_DMD(1:nx*ny),[nx ny])  + ux_mean;
    pcolor(xx,yy,real(u_x_DMD)); shading interp, axis equal tight
    clim([-100 250])
    set(gca,'YTick',[],'XTick',[]); box on
    title(['$\tilde{u}_\mathrm{DMD}^{(' num2str(ti) ')}(\mathbf{x})$'],'Interpreter','latex')

    nexttile(k+3*nt_plot)
    u_y_DMD = reshape(q_DMD(nx*ny+1:end),[nx ny]) + uy_mean;
    pcolor(xx,yy,real(u_y_DMD)); shading interp, axis equal tight
    clim(50*[-1 1])
    set(gca,'YTick',[],'XTick',[]); box on
    title(['$\tilde{v}_\mathrm{DMD}^{(' num2str(ti) ')}(\mathbf{x})$'],'Interpreter','latex')
end
