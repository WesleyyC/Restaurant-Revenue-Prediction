
trainData = csvread('str_num_train.csv',1,0);
trainFeatures = trainData(:,2:end-1);
trainRevenue = trainData(:, end:end);

testData = csvread('str_num_test.csv',1,0);
testFeatures=testData(:,2:end);

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
Ensemble = fitensemble(trainFeatures, trainRevenue,'Bag', 1000, 'Tree', 'Type', 'Regression','FResample',0.25);

CVensembler = crossval(Ensemble, 'KFold', kfold);
err=sqrt(kfoldLoss(CVensembler));

% end
