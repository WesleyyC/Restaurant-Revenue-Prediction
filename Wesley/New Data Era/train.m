
trainData = csvread('str_num_train.csv',1,0);
trainFeatures = trainData(:,2:end-1);
trainRevenue = trainData(:, end:end);

testData = csvread('str_num_test.csv',1,1);
testFeatures=testData;

%% Handle Outline

for i = 1:length(trainRevenue)
    
    if trainRevenue(i)>1e7
        trainRevenue(i)=1e7;
    end
    
end

weight=ones(size(trainRevenue))*10;

weight([17,76,100])=1;



%%
%kfold=5;
%err=zeros([1,10]);
%for m=1:10
%Ensemble = fitensemble(x2fx(trainFeatures, 'linear'), trainRevenue,'Bag', 600, 'Tree', 'Type', 'Regression','FResample', 1/2,'LearnRate',0.8);
%CVensembler = crossval(Ensemble, 'KFold', kfold);
%err=kfoldLoss(CVensembler);
%end


%%
%
%err=Inf([1,N]);

%for n = 1:N
    BaggedEnsemble = TreeBagger(600,trainFeatures,trainRevenue,'OOBVarImp','On','Method', 'regression', 'MaxNumSplits',1,'MinLeafSize',20);
    oobErrorBaggedEnsemble = oobError(BaggedEnsemble);
    %plot(oobErrorBaggedEnsemble)

   % err(n)=sqrt(oobErrorBaggedEnsemble(end))   ;
%end