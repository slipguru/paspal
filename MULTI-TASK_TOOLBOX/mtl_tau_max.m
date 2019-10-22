function tau_max = mtl_tau_max(X,Y)
%MTL_TAU_MAX estimates maximum value for l1 parameter
% 
% [TAU_MAX] = MTL_TAU_MAX(X,Y) estimates maximum value for l1 parameter 
%   for training set (X,Y). 
%   X={X_1,...,X_T} and Y={Y_1,...,Y_T} are cells containing the input 
%   matrices and output vectors, respectively, for each task. 
%   (X{t},Y{t}) is the training set for the t-th task. Y{t} is the N_tx1 
%   label vector, X{t} is the (N_t)X(N_t*D) input matrix for task t


tau_max = 0;
for t = 1:length(X);
    tau_max = tau_max + norm(X{t}'*Y{t})/length(Y{t});
end