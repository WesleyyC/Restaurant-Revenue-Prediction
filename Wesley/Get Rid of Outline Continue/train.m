
trainData = csvread('str_num_train.csv',1,0);
trainFeatures = trainData(:,2:end-1);
trainRevenue = trainData(:, end:end);

testData = csvread('str_num_test.csv',1,0);
testFeatures = testData(:,2:end);

trainFeatures=trainFeatures([1:16,18:75,77:99,101:end]',:);
trainRevenue=trainRevenue([1:16,18:75,77:99,101:end]',:);


%% Handle Outline

for i = 1:length(trainRevenue)
    
    if trainRevenue(i)>1e7
        trainRevenue(i)=1e7;
        i
    end
    
   
end

% outlier = [17,76,100];
% 
% weight = ones(size(trainRevenue));
% weight(outlier)=0.1;

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
trainFeatures1=trainFeatures(lessZero');
trainRevenue1=trainRevenue(lessZero');

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
Ensemble0 = fitensemble(trainFeatures0, trainRevenue0,'Bag', 1000, 'Tree', 'Type', 'Regression');
CVensembler0 = crossval(Ensemble0, 'KFold', kfold);
err0=sqrt(kfoldLoss(CVensembler0))

%%
kfold=10;
Ensemble1 = fitensemble(trainFeatures1, trainRevenue1,'Bag', 1000, 'Tree', 'Type', 'Regression');
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
