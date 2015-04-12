%% Bagging with combination

clear

%% Data Read
filename6Train = 'str_num_train.csv';
data = csvread(filename6Train,1,0);
trainFeatures = data(:,2:end-1);
trainRevenue = data(:,end);


%% Bagging
NLearn = 100;
kfold = 10;

% combination base
N_features = 41;
base=1:N_features;
loop_Start = 39;
loop_End=41;
% compare record
err=zeros(1,loop_End-loop_Start+1);

%%
for n = loop_Start:loop_End
    % set min
    min = Inf;  
    % creating combinations
    C = nchoosek(base,n);
    
    for i = 1:length(C(:,1))
        
        predictor = fitensemble(x2fx(trainFeatures(:,C(i,:)),'interaction'), trainRevenue,  ...
        'Bag', NLearn, 'Tree', 'Type', 'Regression');
        CVensembler = crossval(predictor, 'KFold', kfold);
        mse = sqrt(kfoldLoss(CVensembler))
        if mse < min
            err(n-loop_Start+1) = mse;
            vec(n-loop_Start+1,:) = C(i,:);
            min = mse;
        end
    end
    
    
end

%%
predictor_apply = fitensemble(x2fx(trainFeatures(:,C(i,:)),'interaction'), trainRevenue,  ...
'Bag', NLearn, 'Tree', 'Type', 'Regression');

