%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Classical method: Dynamic Mode Decomposition (DMD)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% 9/19/2025
% OTS <oschmidt@ucsd.edu>

function [Psi,S,L,omega] = dmd(Q,dt,n_DMD)
    % Exact DMD with robust truncation
    Q1      = Q(:,1:end-1);
    Q2      = Q(:,2:end  );

    % Economy SVD then truncate
    [U,Smat,V] = svd(Q1,'econ');
    s          = diag(Smat);
    tol        = max(numel(s)*eps(s(1)), 1e-12*s(1));
    r_eff      = sum(s > tol);
    r          = min([n_DMD, r_eff, size(U,2)]);
    U          = U(:,1:r);
    s          = s(1:r);
    V          = V(:,1:r);
    % Scale V by inverse singular values
    V_scaled   = V ./ (s.');

    % Low-rank projected operator and eigendecomposition
    Atilde     = (U') * Q2 * V_scaled;
    [W,D]      = eig(Atilde);
    L          = diag(D);              % discrete-time eigenvalues
    Psi        = Q2 * V_scaled * W;    % DMD modes
    omega      = log(L)/dt;            % continuous-time eigenvalues
    S          = s;                    % retained singular values

    % Sort modes by |lambda|
    [~,idx] = sort(abs(L),'descend');
    L       = L(idx);
    omega   = omega(idx);
    Psi     = Psi(:,idx);
end