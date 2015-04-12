filename6Train = '~/WillB/Brandeis/Statistical Machine Learning/TermProject_2015/str_num_train.csv';
data = csvread(filename6Train,1,0);
%trainFeatures = data(1:120,2:end-1);
trainFeatures = data(1:120,2:end-1)
%normalized feautres
%trainFeatures = zscore(trainFeatures);
%trainRevenue = data(1:120, end:end);
trainRevenue = data(1:120, end:end);
linearModel = fitlm(trainFeatures,trainRevenue);
%, 'Categorical', [2,3,4]
%predict
testFeatures = data(121:end,2:end-1);
%testFeatures = zscore(testFeatures);
testRevenue = data(121:end, end:end);
[n m] = size(testRevenue);
yhat = [ones(n,1), testFeatures]*linearModel.Coefficients.Estimate;
rme = ((testRevenue - yhat)'*(testRevenue - yhat)/n)^0.5;
fprintf('rmse for basic linear regression: %.2f\n', rme);

%regression tree
tree = fitrtree(trainFeatures, trainRevenue);
L = loss(tree, testFeatures, testRevenue);
fprintf('rmse for regression tree: %.2f\n', L^0.5);

%bagging
colors = 'rbcmg';
NLearn = 12000;
kfold = 10;
isCategorical = [zeros(1,1);ones(size(trainFeatures,2)-1,1)];
%predictor = fitensemble(trainFeatures, trainRevenue,...
     %'Bag', NLearn, 'Tree', 'Type', 'Regression', 'FResample', 0.01);
%CVensembler = crossval(predictor, 'KFold', kfold);
%hold on;
%ens = fitensemble(trainFeatures,trainRevenue,'bag',10000    ,'Tree',...
    %'type','regression', 'Resample', 'On');
%b = TreeBagger(100,trainFeatures,trainRevenue, 'Method','R', 'CategoricalPredictors', find(isCategorical == 1), 'OOBPred','On');
%L = loss(predictor, testFeatures, testRevenue);
%fprintf('rmse for bagging: %.2f\n', L^0.5);

%oldFeatures = trainFeatures(:,2:2);
%newFeatures = ones(size(trainFeatures(:,2:2)));
   %{ 65
   % 66
   % 68
    %69
    %71
    %75
    %77
    %83
    %84
    %85
   %128
   %129

%for i = 1:numel(oldFeatures)
  %  if oldFeatures(i) == 65
   %         newFeatures(i) = 0;
   % elseif oldFeatures(i) == 66
   %         newFeatures(i) = 1;
   % elseif oldFeatures(i) == 68
   %         newFeatures(i) = 2;
   % elseif oldFeatures(i) == 69
    %        newFeatures(i) = 3;
    %elseif oldFeatures(i) == 71
     %       newFeatures(i) = 4;
    %elseif oldFeatures(i) == 75
     %       newFeatures(i) = 5;
    %elseif oldFeatures(i) == 77
     %       newFeatures(i) = 6;
    %elseif oldFeatures(i) == 83
    %        newFeatures(i) = 7;
    %elseif oldFeatures(i) == 84
     %       newFeatures(i) = 8;
    %elseif oldFeatures(i) == 85
     %       newFeatures(i) = 9;
    %elseif oldFeatures(i) == 128
     %       newFeatures(i) = 10;
    %elseif oldFeatures(i) == 129
     %       newFeatures(i) = 11;
    %end
%end

%trainFeatures(:,2:2) = newFeatures;

%fileName = '~/WillB/Brandeis/Statistical Machine Learning/TermProject_2015/trainMonthName.csv'
%data = csvread(fileName,1,0);

%months = data(:,5:5);
%seasons = ones(size(data(:,5:5)));
%%   if months(i) == 3 || months(i) == 4 || months(i) == 5
  %      seasons(i) = 0;
   %%    seasons(i) = 1;
    %elseif months(i) == 9 || months(i) == 10 || months(i) == 11
     %   seasons(i) = 2;
    %else
    %    seasons(i) = 3;
    %end
%end

%data(:,5:5) = seasons;
%%trainFeatures = data(1:120,2:end-1);
%trainRevenue = data(1:120, end:end);
%testFeatures = data(121:end,2:end-1);
%testRevenue = data(121:end, end:end);

%linearModel = fitlm(trainFeatures,trainRevenue);
%[n m] = size(testRevenue);
%yhat = [ones(n,1), testFeatures]*linearModel.Coefficients.Estimate;
%rme = ((testRevenue - yhat)'*(testRevenue - yhat)/n)^0.5;
%fprintf('rmse for basic linear regression: %.2f\n', rme);


n = 15;%17;
%b = nnz(A);
min = 100000000000;
vec = [];
bestModel = [];
for b = 1:8%1:5
    M = 0:ones(1,n)*pow2(n-1:-1:0)';
    z = rem(floor(M(:)*pow2(1-n:0)),2);
    out = z(sum(z,2) == b,:);
    for i = 1:size(out,1)
        logVec = out(i,:);
        catIndexVec = find(logVec == 1);
        %RegTreeEns = fitensemble(trainFeatures, log(trainRevenue),'LSBoost',300,RegTreeTemp, 'CategoricalPredictors', catIndexVec);
        %L = loss(RegTreeEns, testFeatures, log(testRevenue));
        % bagger = fitensemble(trainFeatures, trainRevenue,  ...
        %    'Bag', 200, 'Tree', 'Type', 'Regression','FResample', 0.1, 'CategoricalPredictors', catIndexVec);
       % L = loss(bagger, testFeatures, testRevenue);
        bagger = fitensemble(x2fx(trainFeatures, 'interaction'), trainRevenue,  ...
            'Bag', 100, 'Tree', 'Type', 'Regression','FResample', 0.3, 'CategoricalPredictors', catIndexVec);
        L = loss(bagger, x2fx(testFeatures, 'interaction'), testRevenue);
        %tree = fitrtree(trainFeatures, trainRevenue, 'CategoricalPredictors',catIndexVec);
        %L = loss(tree, testFeatures, testRevenue);
        mse = L^0.5;
        fprintf('mse for current trial: %.2f\n', mse);
        if mse < min
            min = mse;
            vec = catIndexVec;
            %bestModel = tree;
            bestModel = bagger;
            %bestModel = RegTreeEns;
        end
    end
end

%filename6Train = '~/WillB/Brandeis/Statistical Machine Learning/TermProject_2015/str_num_test.csv';
%data = csvread(filename6Train,1,0);
%features = data(:,2:end);
%csvwrite(filename,M)
%pred = predict(bestModel, features);
%M = [data(:,1:1), pred];
%filename = '~/WillB/Brandeis/Statistical Machine Learning/TermProject_2015/subResults201404110844.csv';
