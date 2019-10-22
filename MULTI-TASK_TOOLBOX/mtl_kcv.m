function [cv_output,model] = mtl_kcv(X,Y,varargin)
%MTL_KCV Parameters choice through cross validation for the mtl_algorithm 
%(variable selector) followed by Regularized Least Squares (for debiasing).
%   CV_OUTPUT = MTL_KCV(X,Y) Given T training set, ({X_1,...,X_T},{Y_1,...,Y_T}),
%   performs leave-one-out cross validation and finds the optimal 
%   L1 parameter among values in tau_values for the multitask learning 
%   Task-wise L1L2 is used for selection whereas the regression coefficients 
%   are evaluated on the selected variables via rls task-wise. 
%   (X{t},Y{t}) is the training set for the t-th task. Y{t} is the N_tx1 
%   label vector, X{t} is the (N_t)X(N_t*D) input matrix for task t.
% 
%   [CV_OUTPUT,MODEL] = MTL_KCV(X,Y) Also returns the estimated model
% 
%   MTL_KCV(...,'PropertyName',PropertyValue,...) sets properties to the
%   specified property values.
%       -'L1_n_par': number of values for the L1 parameter (default is 100) 
%       -'L1_max_par': maximum value for the L1 parameter (default is
%        chosen automatically, see paper)
%       -'L1_min_par': minimum value for the L1 parameter (default is
%        chosen automatically)
%       -'L1_pars': vector of values for the L1 parameter. When not specified,
%        100 values are chosen automatically, or according with
%        'L1_n_par','L1_max_par', and 'L1_min_par'.
%       -'RLS_pars': values for RLS parameter. When not specified,
%        50 values are chosen automatically.
%       -'smooth_par': value of the smoothing parameter (default is 0)
%       -'err_type': 'regr'(deafult) for regression, 'class' for 
%        classfication
%       -'protocol': (default is 'two_steps') if 'one_step' evaluates error
%        of the model learned via MTL without RLS-debiasing; if
%        'two_steps' evaluates error of the model learned via MTL with 
%        RLS-debiasing; if 'both' evaluates both.
%       -'offset': (default is true) add unpenalized offset to the model.
%       -'K': specify number K of folds in in  K-fold cross-validation.
%        If K=0 or K=length(Y) it performs LOO cross-validation
%       -'rand_split': if false (default) perfoms a deterministic split of
%       the data, if true the split is random.
%       -'plot': (default is false) if true plots training,  validation 
%        errors, and number of selected variables vs the L1 parameter.
%
%   CV_OUTPUT's fields
%	-sparsity(array of double): number of selected features for each value of the L1 parameter
% 	-selected_all(cell array): indeces of the selected features for each value of the L1 parameter.
%   if 'protocol'=='one_step':
%       -tau_opt_1step(double): L1 parameter minimizing the K-fold
%        cross-validation error for the 1-step framework (MTL only)
%       -err_KCV_1step(array of double): cross-validation error on validation set for
%        the 1-step framework, for each task 
%       -err_train_1step(array of double): cross-validation error on training set for the
%        1-step framework, for each task 
%   if 'protocol'=='two_steps':
%       -tau_opt_2steps(double): L1 parameter minimizing the K-fold
%        cross-validation error for the 2-steps framework (MTL and RLS)
%       -lambda_opt_2steps(double): RLS parameter minimizing the K-fold
%        cross-validation error for the 2-steps framework
%       -err_KCV_2steps(2d array of double): cross-validation error on validation set for the
%        2-steps framework, for each task
%       -err_train_2steps(2d array of double): cross-validation error on training set for the
%        2-steps framework, for each task
%  if 'protocol'=='both': has both of the above sets of fields.
%
%   MODEL's fields
%   -offset: array of offsets to be added to the estimated model or each
%    task
%   if 'protocol'=='one_step':
%       -selected_1step: indexes of the selected features for the optimal
%        parameters  for the 1-step framework
%       -beta_1step: coefficient matrix for the optimal parameters for the
%        1-step framework; each column corresponds to a task
%   if 'protocol'=='two_steps':
%       -selected_2steps: indexes of the selected features for the optimal
%        parameters for the 2-steps framework
%       -beta_2steps: coefficient matrix for the optimal parameters for the 2-steps framework; 
%        each column corresponds to a task.
%  if 'protocol'=='both': has both of the above sets of fields.
%
%   See also MTL_ALGORITHM, MTL_REGPATH, RLS_ALGORITHM
%

if nargin<2, error('too few input!'), end

% DEFAULT PARAMETRS
err_type = 'regr';
smooth_par = 0;
ntau = 100;
tau_min = [];
tau_max = []; 
tau_values = [];
lambda_values = [];
K = 0;
split = false;
center = true;
plotting = false;    

% OPTIONAL PARAMETERS
args = varargin;
nargs = length(args);
for i=1:2:nargs
    switch args{i},
		case 'L1_pars'
            tau_values = args{i+1};
		case 'L1_min_par'
            tau_min = args{i+1};
		case 'L1_max_par'
            tau_max = args{i+1};
		case 'L1_n_par'
            ntau = args{i+1};
 		case 'RLS_pars'
            lambda_values = args{i+1};
		case 'err_type'
            err_type = args{i+1};
		case 'smooth_par'
            smooth_par = args{i+1};
		case 'K'
            K = args{i+1};
		case 'rand_split'
            split = args{i+1};
		case 'plot'
            plotting = args{i+1};
		case 'protocol'
            if strcmp(args{i+1},'one_step');
                withRLS = false;
                woRLS = true;
            elseif strcmp(args{i+1},'two_steps');
                withRLS = true;
                woRLS = false;
            elseif strcmp(args{i+1},'both');
                withRLS = true;
                woRLS = true;
            else
                error('Unknown protocol!!!')
            end
    end
end

T = length(X);
Xtmp = cell(T,1);
Ytmp = cell(T,1);
for i_task = 1:T;
    [Xtmp{i_task},Ytmp{i_task}] = centering(X{i_task},Y{i_task},center);
end

if isempty(tau_values);
    if isempty(tau_max);
        tau_max = mtl_tau_max(Xtmp,Ytmp);
    end
    if isempty(tau_min);
        tau_min = tau_max/100;
    end
    tau_values = [tau_min tau_min*((tau_max/tau_min)^(1/(ntau-1))).^(1:(ntau-1))]; %geometric series.
else
    ntau = length(tau_values);
end
if isempty(lambda_values);
    sigma = zeros(T,1);
    for i_task = 1:T;
        sigma(i_task) = normest(Xtmp{i_task}*Xtmp{i_task}');
    end
    lambda_values = sigma*(10.^(-9.8:.2:0));
end
clear Xtmp Ytmp

sets = cell(T,1);
for i_task = 1:T;
    sets{i_task} = splitting(Y{i_task},K,split); %splits the training set in K subsets
end

% initialization
if woRLS
    err_KCV = zeros(length(sets),T,ntau);
    err_train = zeros(length(sets),T,ntau);
end
if withRLS
    err_KCV2 = zeros(length(sets),T,ntau,length(lambda_values));
    err_train2 = zeros(length(sets),T,ntau,length(lambda_values));
end
selected = cell(length(sets),1);
sparsity = zeros(length(sets),ntau);

for i = 1:length(sets{1});
    Xtr = cell(T,1);
    Ytr = cell(T,1);
    Xts = cell(T,1);
    Yts = cell(T,1);
    meanY = zeros(T,1);
    meanX = cell(T,1);
    for i_task = 1:T;
        ind = setdiff(1:length(Y{i_task}),sets{i_task}{i}); %indexes of training set    
        % normalization
        [Xtr{i_task},Ytr{i_task},meanX{i_task},meanY(i_task)] = normalization(X{i_task}(ind,:),Y{i_task}(ind),center);
        Xts{i_task} = X{i_task}(sets{i_task}{i},:);
        Yts{i_task} = Y{i_task}(sets{i_task}{i});
    end
    % evaluate all betas for all tau_values concurrently
    [selected{i}, sparsity(i,:),k,beta] = mtl_regpath(Xtr,Ytr,tau_values,'smooth_par',smooth_par); 
    selected{i} = selected{i}~=0;
    % for each value of the L1 parameter, use the mtl solution for
    % selection and train rls on the selected features, then evaluate error
    % on validation set (err_KCV)
    for t = 1:ntau;
        for i_task = 1:T;
            
            if woRLS
                offset = meanY(i_task)-meanX{i_task}*beta{t}(:,i_task);
                err_KCV(i,i_task,t) = linear_test(Xts{i_task},Yts{i_task},beta{t}(:,i_task),err_type,offset);       
                err_train(i,i_task,t) = linear_test(Xtr{i_task},Ytr{i_task},beta{t}(:,i_task),err_type,offset);       
            end
            if withRLS
                beta_rls = rls_regpath(Xtr{i_task}(:,selected{i}(:,t)),Ytr{i_task},lambda_values);
                for l = 1:length(lambda_values);
                    offset = meanY(i_task)-meanX{i_task}(selected{i}(:,t))*beta_rls{l};
                    err_KCV2(i,i_task,t,l) = linear_test(Xts{i_task}(:,selected{i}(:,t)),Yts{i_task},beta_rls{l},err_type,offset);       
                    err_train2(i,i_task,t,l) = linear_test(Xtr{i_task}(:,selected{i}(:,t)),Ytr{i_task},beta_rls{l},err_type,offset);       
                end    
            end
            
        end
    end
    
end

cv_output.selected_all = selected;
cv_output.sparsity = mean(sparsity);
clear selected sparsity

if woRLS
    % evaluate avg. error over the splits
    err_KCV = reshape(mean(err_KCV,1),T,ntau,1);
    err_train = reshape(mean(err_train,1),T,ntau,1);

    err_KCV_mean = reshape(mean(err_KCV,1),ntau,1);
    
    % find L1 parameter minimizing the error
    t_opt = find(err_KCV_mean==min(err_KCV_mean),1,'last');
    cv_output.tau_opt_1step = tau_values(t_opt);
    cv_output.err_KCV_1step = err_KCV;
    cv_output.err_train_1step = err_train;
end

if withRLS
    % evaluate avg. error over the splits
    err_KCV2 = reshape(mean(err_KCV2,1),T,ntau,length(lambda_values));
    err_train2 = reshape(mean(err_train2,1),T,ntau,length(lambda_values));

    % for each value of the L1 parameter, find rls parameter minimizing the
    % error
    lambda_opt = zeros(T,ntau);
    err_KCV_opt2 = zeros(T,ntau);
    err_train_opt2 = zeros(T,ntau);

    for i_task = 1:T;
        for t = 1:ntau;
            l_opt = find(err_KCV2(i_task,t,:)==min(err_KCV2(i_task,t,:)),1,'last');
            lambda_opt(i_task,t) = lambda_values(l_opt);
            err_KCV_opt2(i_task,t) = err_KCV2(i_task,t,l_opt);
            err_train_opt2(i_task,t) = err_train2(i_task,t,l_opt);
        end
    end
    
    
    % find L1 parameter minimizing the error
    t_opt2 = find(mean(err_KCV_opt2,1)==min(mean(err_KCV_opt2,1)),1,'last');
    cv_output.tau_opt_2steps = tau_values(t_opt2);
    cv_output.lambda_opt_2steps = lambda_opt(:,t_opt2);
    cv_output.err_KCV_2steps = err_KCV2;
    cv_output.err_train_2steps = err_train2;
end

if nargout>1
	meanX = cell(T,1);
    meanY = zeros(T,1); 
    for i_task = 1:T;
        [X{i_task},Y{i_task},meanX{i_task},meanY(i_task)] = centering(X{i_task},Y{i_task},center);
    end

    if woRLS;
        [selected,out1,out2,beta] = mtl_regpath(X,Y,tau_values(t_opt:end),'smooth_par',smooth_par,'all_path',false); 
        model.selected_1step = selected~=0;
        model.beta_1step = beta;
        model.offset_1step = zeros(T,1); 
        for i_task = 1:T;
            model.offset_1step(i_task) = meanY(i_task)-meanX{i_task}*beta(:,i_task);
        end
    end

    if withRLS
        selected = mtl_regpath(X,Y,tau_values(t_opt2:end),'smooth_par',smooth_par,'all_path',false)~=0;
        beta = zeros(length(selected),T);
        for i_task = 1:T;
            beta_rls = rls_algorithm(X{i_task}(:,selected),Y{i_task},lambda_opt(i_task,t_opt2));    
            beta(selected,i_task) = beta_rls;
        end
        model.selected_2steps = selected;
        model.beta_2steps = beta;        
        model.offset_2steps = zeros(T,1); 
        for i_task = 1:T;
            model.offset_2steps(i_task) = meanY(i_task)-meanX{i_task}*beta(:,i_task);
        end

    end
end


if plotting;
    figure('Name','Multi-task learning')
    c = 0;
        c = 0;
    if woRLS
        c = c+1;
        subplot(withRLS+woRLS+1,1,c)
        semilogx(tau_values,mean(err_train),'bs-','MarkerSize',3,'MarkerFaceColor','b'); hold on;
        semilogx(tau_values,mean(err_KCV),'rs-','MarkerSize',3,'MarkerFaceColor','r'); 
        legend('train','validation');
        xlim = get(gca,'Xlim');
        ylim = get(gca,'Ylim');
        semilogx(xlim,repmat(min(mean(err_train)),2,1),'b:')
        semilogx(xlim,repmat(min(mean(err_KCV)),2,1),'r:')
        semilogx(repmat(tau_values(t_opt),2),[ylim(1) min(mean(err_KCV))],'r:')
        title('CV error without RLS');
    end
    if withRLS;
        c = c+1;
        subplot(withRLS+woRLS+1,1,c)
        semilogx(tau_values,mean(err_train_opt2),'bs-','MarkerSize',3,'MarkerFaceColor','b'); hold on;
        semilogx(tau_values,mean(err_KCV_opt2),'rs-','MarkerSize',3,'MarkerFaceColor','r'); 
        legend('train','validation');
        xlim = get(gca,'Xlim');
        ylim = get(gca,'Ylim');
        semilogx(xlim,repmat(min(mean(err_train_opt2)), 2,1),'b:')
        semilogx(xlim,repmat(min(mean(err_KCV_opt2)), 2,1),'r:')
        semilogx(repmat(tau_values(t_opt2),2),[ylim(1) min(mean(err_KCV_opt2))],'r:')
        title('CV error with RLS');
    end
    c = c+1;
    subplot(withRLS+woRLS+1,1,c)
    semilogx(tau_values,cv_output.sparsity,'ks-','MarkerSize',3,'MarkerFaceColor','b'); 
    hold on
    if woRLS
        semilogx([xlim(1) tau_values(t_opt)],repmat(cv_output.sparsity(t_opt),2,1),'r:')
        semilogx(repmat(tau_values(t_opt),2),[ylim(1) cv_output.sparsity(t_opt)],'r:')        
    end
    if withRLS
        semilogx([xlim(1) tau_values(t_opt2)],repmat(cv_output.sparsity(t_opt2),2,1),'r:')
        semilogx(repmat(tau_values(t_opt2),2),[ylim(1) cv_output.sparsity(t_opt2)],'r:')        
    end
    xlabel('\tau');
    title('# of selected variables');
end
