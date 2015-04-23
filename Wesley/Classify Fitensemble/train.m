%% Classify Fitensemble

clear

%% Data Read

trainData = csvread('str_num_train.csv',1,0);
trainFeatures = trainData(:,4:end-1);   %ignore open date and city name
trainRevenue = trainData(:,end);
actualRevenue = trainRevenue;
trainRevenue = round(trainRevenue/1e6)*1e6;

testFeatures = csvread('str_num_test.csv',1,3);

%% Convert to 

2306598
tree = fitctree(trainFeatures,trainRevenue);

%% Boosting

NLearn=300;
kfold=5;

%good
%1950283.3
Ensemble = fitensemble(x2fx(trainFeatures, 'quadratic'),trainRevenue,'AdaBoostM2',NLearn,'Tree');
CVensembler = crossval(Ensemble, 'KFold', kfold);
plot(kfoldLoss(CVensembler, 'Mode', 'cumulative'));


%%