% Handloe Train

current_result = csvread('Workbook1.csv',1,1);

trainData = csvread('str_num_train.csv',1,0);
trainFeatures = trainData(:,[32,2,27,21,5,25,6,20,22,29,10,24]);
trainRevenue = trainData(:, end:end);

testData = csvread('str_num_test.csv',1,0);
testFeatures=testData(:,[32,2,27,21,5,25,6,20,22,29,10,24]);

trainFeatures=[trainFeatures',testFeatures']';
trainRevenue=[trainRevenue',current_result']';
weight=ones(size(trainRevenue));
moreWeight = 1:length(trainData);
weight(moreWeight')=weight(moreWeight')*10;



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

%% Get The Zero One

lessZero=[];
moreZero=[];

for i=1:length(trainFeatures)
    numZero = sum(trainFeatures(i,:)==0);
    if(numZero>=4)
        moreZero=[moreZero,i];
    else
        lessZero=[lessZero,i];
    end
end

trainFeatures0=trainFeatures(moreZero');
trainRevenue0=trainRevenue(moreZero');
weight0=weight(moreZero');
trainFeatures1=trainFeatures(lessZero');
trainRevenue1=trainRevenue(lessZero');
weight1=weight(lessZero');
%%
lessZero=[];
moreZero=[];

for i=1:length(testFeatures)
    numZero = sum(testFeatures(i,:)==0);
    if(numZero>=4)
        moreZero=[moreZero,i];
    else
        lessZero=[lessZero,i];
    end
end   

testFeatures0=testFeatures(moreZero');
testFeatures1=testFeatures(lessZero');


%%
kfold=10;
Ensemble0 = fitensemble(trainFeatures0, trainRevenue0,'Bag', 1000, 'Tree', 'Type', 'Regression','Weight',weight0);
CVensembler0 = crossval(Ensemble0, 'KFold', kfold);
err0=sqrt(kfoldLoss(CVensembler0))

%%
kfold=10;
Ensemble1 = fitensemble(trainFeatures1, trainRevenue1,'Bag', 1000, 'Tree', 'Type', 'Regression','Weight',weight1);
CVensembler1 = crossval(Ensemble1, 'KFold', kfold);
err1=sqrt(kfoldLoss(CVensembler1))

%% output
output0=predict(Ensemble0,testFeatures0);
output1=predict(Ensemble1,testFeatures1);
moreZero=moreZero-1;
lessZero=lessZero-1;
output0=[moreZero',output0];
output1=[lessZero',output1];
output=[output0',output1']';
output=sortrows(output);
output(:,2)=int64(output(:,2));



