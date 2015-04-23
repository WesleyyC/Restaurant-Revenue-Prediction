
trainData = csvread('str_num_train.csv',1,0);
trainFeatures = [trainData(:,2),trainData(:,4:end-1)];
trainRevenue = trainData(:, end:end);

testData = csvread('str_num_test.csv',1,0); 
testFeatures=[testData(:,2),testData(:,4:end)];
%% Handle Outline

outlineFlag=zeros(size(trainRevenue));

for i = 1:length(trainRevenue)
    
    if trainRevenue(i)>10e6
        outlineFlag(i)=1;
    end
    
end


%%
kfold=5;
%err=zeros([1,10]);
%for m=1:10
Ensemble = fitensemble([trainFeatures,outlineFlag], trainRevenue,'Bag', 1000, 'Tree', 'Type', 'Regression');
CVensembler = crossval(Ensemble, 'KFold', kfold);
err=sqrt(kfoldLoss(CVensembler));
%end

%% model for flag

Classification=fitensemble(trainFeatures,outlineFlag,'Bag',1000,'Tree','Type','Classification');
CVclassification = crossval(Classification, 'KFold', kfold);
err2=kfoldLoss(CVclassification);

%%
outlineFlag=predict(Classification,testFeatures);
output=int64(predict(Ensemble,[testFeatures,outlineFlag]));