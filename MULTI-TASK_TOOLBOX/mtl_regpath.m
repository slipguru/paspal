function [selected,sparsity,k,beta] = mtl_regpath(X,Y,tau_values,varargin)
% MTL_REGPATH acceleration of mtl through cascade of mtl w. decreasing 
% values of TAU_VALUES
% 
% [SELECTED] = MTL_REGPATH(X,Y,TAU_VALUES) for each value in TAU_VALUES
%   evaluates mtl solution with l2 parameter 0, and builds cell of indexes 
%   of selected features. 
%   X={X_1,...,X_T} and Y={Y_1,...,Y_T} are cells containing the input 
%   matrices and output vectors, respectively, for each task. 
%   (X{t},Y{t}) is the training set for the t-th task. Y{t} is the N_tx1 
%   label vector, X{t} is the (N_t)X(N_t*D) input matrix for task t

% 
% [SELECTED,SPARSITY] = MTL_REGPATH(X,Y,TAU_VALUES) also returns a vector
%   with the number of selected features for each value in TAU_VALUES
% 
% [SELECTED,SPARSITY,K] = MTL_REGPATH(X,Y,TAU_VALUES) also returns a 
%   vector with the number of kations for each value in TAU_VALUES
% 
% [SELECTED,SPARSITY,K,BETA] = MTL_REGPATH(X,Y,TAU_VALUES) also returns 
%  the coefficients matrix (DxN_t) for each value in TAU_VALUES
% 
% MTL_REGPATH(...,'PropertyName',PropertyValue,...) sets properties to the
%   specified property values.
%       -'smooth_par': (default is 0) sets l2 parameter equal to MU_FACT*step_size
%       -'max_iter': (default is 1e5) maximum number of iterations
%       -'tolerance': (default is 1e-6) tolerance for stopping the iterations.
%       -'all_path': (default is true) if true evaluates all regularization
%        path with given tolerance. Otherwise evaluates solutions for larger 
%        vaues of tau with looser tolerance
 
   
if nargin<3; error('too few inputs!'); end

% DEFAULT PARAMETERS
smooth_par = 0;
kmax = 1e5;
tol = 1e-6;
all_path = true;

% OPTIONAL PARAMETERS
args = varargin;
nargs = length(args);
for i=1:2:nargs
    switch args{i},
		case 'smooth_par'
            smooth_par = args{i+1};
		case 'max_iter'
            kmax = args{i+1};
		case 'tolerance'
            tol = args{i+1};
		case 'all_path'
            all_path = args{i+1};
    end
end

ntau = length(tau_values);
ntasks = length(X); % number of tasks

% if interested in just the first value of TAU_VALUES, evaluates solutions
% for larger vaues of tau with looser tolerance
tol = ones(ntau,1).*tol;
if ~all_path;
    tol(2:end) = tol(2:end).*100;
end


d = size(X{1},2); %number of input variables
eig_max = zeros(ntasks,1);
for i_task = 1:ntasks;
    eig_max(i_task) = normest(X{i_task}*X{i_task}')/size(X{i_task},1);% max eigenvalue of input matrix for task t
end
sigma0 = max(eig_max);

ntot = 0;
for i_task = 1:ntasks;
    ntot = ntot + length(Y{i_task});
end

% the initialization vector beta0 is set equal to the least squares
% solution with independent tasks
beta0 = zeros(ntasks*d,1);


%initialization
beta = cell(ntau,1);
sparsity = zeros(ntau,1);
selected = zeros(d,ntau);
sparsity_prev = 0;

for tau = 1:ntau;

    % when mu=0, if for larger value of tau l1l2 selected less than n 
    % variables, then keep running mtl-l1l2 for smaller values, 
    % o.w. take rls solution
    if and(smooth_par==0,sparsity_prev>=min(ntot,d));
        selected(:,ntau+1-t) = ones(d,1)~=0; % selected variables
        sparsity(ntau+1-t) = d; % number of selected variables
    else
        [beta_tmp,k] = mtl_algorithm(X,Y,tau_values(ntau+1-tau),smooth_par,beta0,sigma0,kmax,tol(ntau+1-tau)); 
        beta0 = beta_tmp; %re-initialize
        beta{ntau+1-tau} = reshape(beta_tmp,d,ntasks);
        sparsity_prev = sparsity(ntau+1-tau);
        % find selected variables
        beta_tot = sum(beta{ntau+1-tau},2); %for each variable j sum the beta_js over all tasks
        selected(:,ntau+1-tau) = beta_tot~=0; %variables selected with tau_values(tau)
        sparsity(ntau+1-tau) = sum(beta_tot~=0); %number of variables selected with tau_values(tau)
    end    
end


if ~all_path;
    beta = beta{1};
    selected = selected(:,1);
    sparsity = sparsity(1);
end
