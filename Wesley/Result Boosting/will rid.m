% Handloe Train

current_result = csvread('Workbook1.csv',1,1);

trainData = csvread('str_num_train.csv',1,0);
trainFeatures = trainData(:,[32,2,27,21,5,25,6,20,22,29,10]);
trainRevenue = trainData(:, end:end);

%% get variance and create outlier flag
stdDev = var(trainRevenue)^0.5;
outlierFlag = zeros(size(trainRevenue));
for i = 1:length(trainRevenue)
     
     if trainRevenue(i)> stdDev + mean(trainRevenue) || trainRevenue(i) < mean(trainRevenue) - stdDev
         outlierFlag(i) = 1;
     end
     
end

%% build model to predict outlier
bag = fitensemble(trainFeatures,outlierFlag,'Bag',200,'Tree',...
    'Type','Classification');
trainFeatures = [trainFeatures, outlierFlag];


%%

testData = csvread('str_num_test.csv',1,0);
testFeatures=testData(:,[32,2,27,21,5,25,6,20,22,29,10]);
testOutlierFlag = predict(bag ,testFeatures);
testFeatures = [testFeatures, testOutlierFlag];

%trainFeatures=[trainFeatures',testFeatures']';
%trainRevenue=[trainRevenue',current_result']';
%% Handle Outline

% for i = 1:length(trainRevenue)
%     
%     if trainRevenue(i)>1e7
%         trainRevenue(i)=1e7;
%     end
%     
% end

%trainFeatures(:,2)=trainFeatures(:,2)-1900;
%testFeatures(:,2)=testFeatures(:,2)-1900;

%%

Ensemble = fitensemble(trainFeatures, trainRevenue, 'Bag', 600, 'Tree', 'Type', 'Regression');
pred = predict(Ensemble, testFeatures);

%% Output file
file = 'str_num_test.csv';
data = csvread(file,1,0);
M = [data(:,1:1), pred];
filename = 'abstract500.csv';
csvwrite(filename,M);



