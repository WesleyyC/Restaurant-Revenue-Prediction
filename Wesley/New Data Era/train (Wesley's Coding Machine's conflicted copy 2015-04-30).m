
trainData = csvread('str_num_train.csv',1,0);
trainFeatures = trainData(:,2:end-1);
trainRevenue = trainData(:, end:end);

testData = csvread('str_num_test.csv',1,0);
testFeatures=testData(:,2:end);

%% Replace 0 with NaN

for i = 1:length(trainFeatures(:,1))
    for j = 1:length(trainFeatures(1,:))
        if trainFeatures(i,j)==0
            trainFeatures(i,j)=NaN;
        end
    end
end

for i = 1:length(testFeatures(:,1))
    for j = 1:length(testFeatures(1,:))
        if testFeatures(i,j)==0
            testFeatures(i,j)=NaN;
        end
    end
end

for i = 1:length(trainFeatures(:,2))
    if isnan(trainFeatures(i,2))
        trainFeatures(i,2) = 0;
    end
end

for i = 1:length(testFeatures(:,2))
    if isnan(testFeatures(i,2))
        testFeatures(i,2) = 0;
    end
end


%% Handle Outline

for i = 1:length(trainRevenue)
    
    if trainRevenue(i)>1e7
        trainRevenue(i)=1e7;
    end
    
    
end


%%
kfold=5;
% err=zeros([1,10]);
% for m=1:10
Ensemble = fitensemble(x2fx(trainFeatures,'linear'), trainRevenue,'LSBoost', 6000, 'Tree','FResample',0.25);

CVensembler = crossval(Ensemble, 'KFold', kfold);
err=sqrt(kfoldLoss(CVensembler))

% end
