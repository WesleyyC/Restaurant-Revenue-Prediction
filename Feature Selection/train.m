
trainData = csvread('str_num_train.csv',1,0);
trainFeatures=trainData(:,[2,3,6,7,22,25,28,33]);
trainRevenue = trainData(:, end);

testData = csvread('str_num_test.csv',1,0);
testFeatures=testData(:,[2,3,6,7,22,25,28,33]);

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
Ensemble = fitensemble(x2fx(trainFeatures,'quadratic'), trainRevenue,'Bag', 5000, 'Tree', 'Type', 'Regression');

CVensembler = crossval(Ensemble, 'KFold', kfold);
plot(kfoldLoss(CVensembler,'mode','cumulative'));
sqrt(kfoldLoss(CVensembler))

% end


