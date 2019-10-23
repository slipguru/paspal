function [beta,n_iter] = mtl_algorithm(X, Y, tau, smooth_par, beta0, sigma0, max_iter,tol)
% MTL_ALGORITHM argmin of least squares error with MTL penalty
% 
% [BETA] = MTL_ALGORITHM(X,Y,TAU) returns the solution of MTL
%   regularization with sparsity parameter TAU and smoothness parameter 0.
%   X={X_1,...,X_T} and Y={Y_1,...,Y_T} are cells containing the input 
%   matrices and output vectors, respectively, for each task. 
%   (X{t},Y{t}) is the training set for the t-th task. Y{t} is the N_tx1 
%   label vector, X{t} is the (N_t)X(N_t*D) input matrix for task t
%   The step size is A*(1+SMOOTH_PAR), where A is the largest among the 
%   largest eigenvalues of all the X{t}'*X{t}/(N{t}*2) and N{t} is the 
%   number of training samples for task t.
%   The algorithm stops when value of the regularized empirical error reaches convergence.
% 
% [BETA] = MTL_ALGORITHM(X,Y,TAU,SMOOTH_PAR) returns the solution of MTL
%   regularization with sparsity parameter TAU and smoothness parameter 
%   A*SMOOTH_PAR.
% 
% [BETA,K] = MTL_ALGORITHM(X,Y,TAU,SMOOTH_PAR) also returns the number of iterations
% 
% [BETA] = MTL_ALGORITHM(X,Y,TAU,SMOOTH_PAR,BETA0) uses BETA0 as 
%   initialization vectors for BETA
% 
% [BETA] = MTL_ALGORITHM(X,Y,TAU,SMOOTH_PAR,BETA0,SIGMA0) sets the 
%   smoothness parameter to SIGMA0*SMOOTH_PAR and and the step size to 
%   SIGMA0*(1+SMOOTH_PAR). If SIGMA0=[], sets the smoothness parameter to 
%   A*SMOOTH_PAR and the step size is A*(1+SMOOTH_PAR).
% 
% [BETA] = MTL_ALGORITHM(X,Y,TAU,SMOOTH_PAR,BETA0,SIGMA0,MAX_ITER) the algorithm stops after
%   MAX_ITER iterations or when regularized empirical error reaches convergence 
%   (default tolerance is 1e-6).
%
%   [...] = L1L2_ALGORITHM(X,Y,TAU,SMOOTH_PAR,BETA0,SIGMA0,MAX_ITER,TOL) uses TOL
%   as tolerance for stopping.

   
    if nargin<3; error('too few inputs!'); end
    if nargin<4; smooth_par = 0; end
    if nargin<5; beta0 = []; end
    if nargin<6; sigma0=[]; end
    if nargin<7; max_iter = 1e5; end
    if nargin<8; tol = 1e-6; end
    if nargin>8; error('too many inputs!'); end

    d = size(X{1},2);
    T = length(X);
    N = zeros(T,1); 
    for task = 1:T
        N(task)  = size(X{task},1);
    end
    
    % if sigma0 is not specified in input, evaluates it
    if isempty(sigma0)
        eig_max = zeros(T,1);
        for task = 1:T
            if isempty(sigma0)
                eig_max(task) = normest(X{task})^2/N(task);% max eigenvalue of input matrix for task t
            end
        end
        sigma0 = max(eig_max); % maximum eigenvalue of matrix Psi'*N*Psi
    end
    
    if(isempty(beta0))
        beta0 = zeros((T*d),1); 
    end

    mu = smooth_par*sigma0; %l2 parameter
    step = sigma0+mu;

    tau_s = tau/step;
    stop = 0;
    mu_s = mu/step;
    XT = cell(T,1);
    XY = cell(T,1);
    for task = 1:T
        XT{task}  = X{task}'./(step*N(task));
        XY{task}  = XT{task}*Y{task};
    end

% initialization
    E_prevs = Inf*ones(10,1);

    beta = beta0;
    h = beta0;
    t = 1;
    Xb = cell(T,1);
    for task = 1:T
        Xb{task} = X{task}*beta((1+(task-1)*d):(task*d));
    end
    Xh = Xb;
    n_iter = 0;
    
% mtl iterations
    while and(n_iter<max_iter,~stop) 
    
        n_iter = n_iter+1;
        beta_prev = beta;                    
        Xb_prev = Xb;
        
        beta_noproj = zeros(T*d,1);
        for task = 1:T
            beta_noproj((1+(task-1)*d):(task*d)) = h((1+(task-1)*d):(task*d)).*(1-mu_s) + XY{task}-XT{task}*Xh{task};
        end
        beta = beta_noproj.*repmat(max(0,1-tau_s./sqrt(sum(reshape(beta_noproj,d,T).^2,2))),T,1);        
        t_new = .5*(1+sqrt(1+4*t^2));
        h = beta + (t-1)/(t_new)*(beta-beta_prev);
        E = 2*tau*sum(sqrt(sum(reshape(beta,d,T).^2,2)));        
        for task = 1:T
            Xb{task} = X{task}*beta((1+(task-1)*d):(task*d));
            Xh{task} = Xb{task}.*(1+ (t-1)/(t_new)) +(1-t)/(t_new).*Xb_prev{task};
            E = E + norm(Xb{task}-Y{task})^2/N(task); 
        end
        if smooth_par>0
            E = E + mu*norm(beta)^2;
        end
        t = t_new;        

        E_prevs(mod(n_iter,10)+1) = E;
        
        if (mean(E_prevs)-E)<mean(E_prevs)*tol; stop =1; end
    end
end