
trainData = csvread('str_num_train.csv',1,0);
trainFeatures=trainData(:,[32,2,27,21,5,25,6,20,22,29,10,24]);
trainRevenue = trainData(:, end);
testData = csvread('str_num_test.csv',1,0);
testFeatures=testData(:,[32,2,27,21,5,25,6,20,22,29,10,24]);

%% Handle Outline

for i = 1:length(trainRevenue)
    
    if trainRevenue(i)>1e7
        trainRevenue(i)=1e7;
    end
end

trainFeatures(:,2)=trainFeatures(:,2)-1900;
testFeatures(:,2)=testFeatures(:,2)-1900;

%%
kfold=5;
% err=zeros([1,10]);
% for m=1:10
Ensemble = fitensemble(trainFeatures, trainRevenue,'Bag', 1000, 'Tree', 'Type', 'Regression');

CVensembler = crossval(Ensemble, 'KFold', kfold);
plot(kfoldLoss(CVensembler,'mode','cumulative'));
sqrt(kfoldLoss(CVensembler))

% end


