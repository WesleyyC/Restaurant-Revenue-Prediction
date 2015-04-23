
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

%% Handle Features
[coeff,score,latent] = pca(trainFeatures);

% sum_latent=sum(latent);
% cum=0;
% latent_cum=zeros([1,length(latent)]);
% for i = 1:length(latent)
%     cum = cum + latent(i);
%     latent_cum(i)=cum/sum_latent;
% end

trainFeatures=trainFeatures*coeff(:,1:5);
testFeatures=testFeatures*coeff(:,1:5);


%%
kfold=5;
%err=zeros([1,20]);
%for m=1:20
Ensemble = fitensemble(x2fx(trainFeatures, 'quadratic'), trainRevenue,'Bag', 500, 'Tree', 'Type', 'Regression','FResample', 1);
CVensembler = crossval(Ensemble, 'KFold', kfold);
err=kfoldLoss(CVensembler)
%end
