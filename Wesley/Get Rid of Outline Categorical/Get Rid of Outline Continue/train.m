
trainData = csvread('str_num_train.csv',1,0);
trainFeatures = [trainData(:,2),trainData(:,4:end-1)];
trainRevenue = trainData(:, end:end);

testData = csvread('str_num_test.csv',1,0);
testFeatures=[testData(:,2),testData(:,4:end)];


%% Handle Outline

for i = 1:length(trainRevenue)
    
    if trainRevenue(i)>1e7
        trainRevenue(i)=1e7;
    end
    
    
end

outlier = [17,76,100];

weight = ones(size(trainRevenue));
weight(outlier)=0.1;

%%
kfold=5;
% err=zeros([1,10]);
% for m=1:10
Ensemble = fitensemble(x2fx(trainFeatures,'interaction'), trainRevenue,'Bag', 1000, 'Tree', 'Type', 'Regression','Weight',weight);

CVensembler = crossval(Ensemble, 'KFold', kfold);
err=sqrt(kfoldLoss(CVensembler))

% end
