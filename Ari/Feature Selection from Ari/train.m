
trainData = csvread('str_num_train.csv',1,1);
trainFeatures = csvread('r_train.csv',1,0);
trainRevenue = trainData(:, end);

testData = csvread('r_test.csv',1,0);
testFeatures=testData;

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
Ensemble = fitensemble(trainFeatures, trainRevenue,'Bag', 600, 'Tree', 'Type', 'Regression');

CVensembler = crossval(Ensemble, 'KFold', kfold);
plot(kfoldLoss(CVensembler,'mode','cumulative'));
sqrt(kfoldLoss(CVensembler))

% end
