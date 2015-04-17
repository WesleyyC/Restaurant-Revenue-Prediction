% Handloe Train

current_result = csvread('submit1e71d.csv',1,1);

trainData = csvread('str_num_train.csv',1,0);
trainFeatures = [trainData(:,2),trainData(:,4:end-1)];
trainRevenue = trainData(:, end:end);

testData = csvread('str_num_test.csv',1,0);
testFeatures=[testData(:,2),testData(:,4:end)];

trainFeatures=[trainFeatures',testFeatures']';
trainRevenue=[trainRevenue',current_result']';
%% Handle Outline

for i = 1:length(trainRevenue)
    
    if trainRevenue(i)>1e7
        trainRevenue(i)=1e7;
    end
    
    
end


%%

Ensemble = fitensemble(x2fx(trainFeatures, 'quadratic'), trainRevenue,'Bag', 600, 'Tree', 'Type', 'Regression');
CVensembler = crossval(Ensemble, 'KFold',5);
plot(kfoldLoss(CVensembler,'mode','cumulative'));


