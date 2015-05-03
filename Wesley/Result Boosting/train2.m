% Handloe Train

current_result = csvread('kill wol.csv',1,1);

trainData = csvread('str_num_train.csv',1,0);
testData = csvread('str_num_test.csv',1,0);
imputeTrain=csvread('trainImpute.csv');
trainData(:,5:41)=imputeTrain;
testData(:,5:41)=csvread('testImpute.csv');

%%
trainFeatures = trainData(:,[32,2,27,21,5,25,6,20,22,29,10,24]);
trainRevenue = trainData(:, end:end);


testFeatures=testData(:,[32,2,27,21,5,25,6,20,22,29,10,24]);

trainFeatures=[trainFeatures',testFeatures']';
trainRevenue=[trainRevenue',current_result']';

weight=ones(size(trainRevenue));
moreWeight = 1:length(trainData);
weight(moreWeight')=weight(moreWeight')*2;
%% Handle Outline

% for i = 1:length(trainRevenue)
%     
%     if trainRevenue(i)>1e7
%         trainRevenue(i)=1e7;
%     end
%     
% end

trainFeatures(:,2)=trainFeatures(:,2)-1900;
testFeatures(:,2)=testFeatures(:,2)-1900;

%%

Ensemble = fitensemble(trainFeatures, trainRevenue, 'Bag', 1000, 'Tree', 'Type', 'Regression','Weight',weight);



