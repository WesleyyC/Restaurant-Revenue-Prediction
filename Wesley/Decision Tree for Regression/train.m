
trainData = csvread('str_num_train.csv',1,0);
trainFeatures = [trainData(:,2),trainData(:,4:end-1)];
trainRevenue = trainData(:, end:end);

testData = csvread('str_num_test.csv',1,0);
testFeatures=[testData(:,2),testData(:,4:end)];


%%

    tree=fitrtree(trainFeatures,trainRevenue);
