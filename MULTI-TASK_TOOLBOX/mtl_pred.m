function pred = mtl_pred(model,Xtest,Ytest,err_type)
%Predicts labels on test set
%   [PRED] = MTL_PRED(MODEL,XTEST) Given MODEL from MTL_KCV predicts labels
%   for test set Xtest. 
%   PRED's fields:
%       -Y_1STEP(if MODEL contains field BETA_1STEP): cell array; each cell
%       contains the predicted labels for each task
%       -Y_2STEPS(if MODEL contains field BETA_2STEPS): cell array; each cell
%       contains the predicted labels for each task
%   [PRED] = MTL_PRED(MODEL,XTEST,YTEST) Given MODEL from MTL_KCV 
%   predicts labels and compute means square errror for test set XTEST. 
%   PRED's fields:
%       if MODEL contains field BETA_1STEP
%           -Y_1STEP: estimated labels
%           -ERR_1STEP: mean square error
%       if MODEL contains field BETA_2STEPS
%           -Y_2STEPS:estimated labels
%           -ERR_2STEPS: mean square error
%   [PRED] = MTL_PRED(MODEL,XTEST,YTEST,ERR_TYPE) if ERR_TYPe='regr' 
%       compute means square errror for test set XTEST. If ERR_TYPE='class'
%       computes classification error. If ERR_TYPE = [w_pos, w_neg],
%       computes weighted classification error.
% 
%   See also MTL_KCV

if nargin==3, err_type = 'regr'; end


T = length(Xtest);
if isfield(model,'beta_1step')
    pred.y_1step = cell(T,1);
end
if isfield(model,'beta_2steps')
    pred.y_2steps = cell(T,1);
end
for i_task = 1:T;
    ntest = size(Xtest{i_task},1);
    if isfield(model,'meanX');
        for i = 1:ntest;
            Xtest{i_task}(i,:) = (Xtest{i_task}(i,:)-model.meanX{i_task});   
        end
    end
    if isfield(model,'stdevsX');
        for i = 1:ntest;
            Xtest{i_task}(i,:) = Xtest{i_task}(i,:)./model.stdevsX{i_task};   
        end
    end
    if ~isfield(model,'meanY');
        model.meanY{i_task} = 0;
    end

    if isfield(model,'beta_1step')
        pred.y_1step{i_task} = Xtest{i_task}*model.beta_1step(:,i_task) + model.meanY{i_task};
    end
    if isfield(model,'beta_2steps')
        pred.y_2steps{i_task} = Xtest{i_task}*model.beta_2steps(:,i_task) + model.meanY{i_task};
    end
end


if nargin>2;    
    if isfield(model,'beta_1step')    
        pred.err_1step = cell(T,1);
    end
    if isfield(model,'beta_2steps')    
        pred.err_2steps = cell(T,1);
    end
    for i_task = 1:T;
        if isequal(err_type,'regr');
            if isfield(model,'beta_1step')    
                pred.err_1step{i_task} = norm(pred.y_1step{i_task}-(Ytest{i_task}))^2/length(Ytest{i_task});
            end
            if isfield(model,'beta_2steps')    
                pred.err_2steps{i_task} = norm(pred.y_2steps{i_task}-(Ytest{i_task}))^2/length(Ytest{i_task});
            end
        else
            npos = sum(Ytest{i_task}>0);
            nneg = sum(Ytest{i_task}<0);
            if strcmp(err_type,'class'); 
                class_fraction = [npos/(npos+nneg) nneg/(npos+nneg)]; 
            else
                class_fraction = err_type;    
            end
            if isfield(model,'beta_1step')
                pred.y_1step{i_task} = sign(pred.y_1step{i_task});
                pred.err_1step{i_task} = 0;
                if npos>0;
                    err_pos = sum((pred.y_1step{i_task}(Ytest{i_task}>0)~=sign(Ytest{i_task}(Ytest{i_task}>0))))/npos;
                    pred.err_1step{i_task} = pred.err_1step{i_task} + err_pos*max(class_fraction(1),nneg==0);
                end
                if nneg>0;
                    err_neg = sum((pred.y_1step{i_task}(Ytest{i_task}<0)~=sign(Ytest{i_task}(Ytest{i_task}<0))))/nneg;
                    pred.err_1step{i_task} = pred.err_1step{i_task} + err_neg*max(class_fraction(2),npos==0);
                end
            end

            if isfield(model,'beta_2steps')
                pred.y_2steps{i_task} = sign(pred.y_2steps{i_task});
                pred.err_2steps{i_task} = 0;
                if npos>0;
                    err_pos = sum((pred.y_2steps{i_task}(Ytest{i_task}>0)~=sign(Ytest{i_task}(Ytest{i_task}>0))))/npos;
                    pred.err_2steps{i_task} = pred.err_2steps{i_task} + err_pos*max(class_fraction(1),nneg==0);
                end
                if nneg>0;
                    err_neg = sum((pred.y_2steps{i_task}(Ytest{i_task}<0)~=sign(Ytest{i_task}(Ytest{i_task}<0))))/nneg;
                    pred.err_2steps{i_task} = pred.err_2steps{i_task} + err_neg*max(class_fraction(2),npos==0);
                end
            end
        end
    end
end