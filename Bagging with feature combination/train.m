%% Data Read
filename6Train = 'str_num_train.csv';
data = csvread(filename6Train,1,0);
trainFeatures = data(:,2:end-1);
trainRevenue = data(:,end);


%% Bagging
NLearn = 500;
kfold = 10;
min = Inf;
vec = [];
bestModel = [];
for b = 2
    % creating combination
    M = 0:ones(1,b)*pow2(b-1:-1:0)';
    z = rem(floor(M(:)*pow2(1-b:0)),2);
    out = z(sum(z,2) == b,:);
    
    for i = 1:size(out,1)
        logVec = out(i,:);
        catIndexVec = find(logVec == 1);
        predictor = fitensemble(trainFeatures, trainRevenue,  ...
            'Bag', NLearn, 'Tree', 'Type', 'Regression','CategoricalPredictors', catIndexVec,'CrossVal','On','KFold', kfold);
        mse = kfoldLoss(predictor)
        if mse < min
            min = mse;
            vec = catIndexVec;
            bestModel = predictor;
        end
    end
    
    
end

