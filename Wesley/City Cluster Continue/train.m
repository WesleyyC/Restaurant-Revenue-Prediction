


%% cluster city type
trainData = csvread('str_num_train.csv',1,0);

train0 = trainData(~ismember(trainData(:,4),[1]),:);
train1 = trainData(~ismember(trainData(:,4),[0]),:);

train0Features = [train0(:,2),train0(:,5:end-1)];
train0Revenue = train0(:, end:end);

train1Features = [train1(:,2),train1(:,5:end-1)];
train1Revenue = train1(:, end:end);

testData = csvread('str_num_test.csv',1,0);
test0 = testData(~ismember(testData(:,4),[1]),:);
test1 = testData(~ismember(testData(:,4),[0]),:);

test0Features = [test0(:,2),test0(:,5:end)];
test1Features = [test1(:,2),test1(:,5:end)];



%% Handle Outline

for i = 1:length(train1Revenue)
    
    if train1Revenue(i)>1e7
        train1Revenue(i)=1e7;
    end
    
end

%% model for train0
%kfold=5;


Ensemble = fitensemble(x2fx(train0Features, 'interaction'), train0Revenue,'Bag', 600, 'Tree', 'Type', 'Regression','FResample', 1/3);
%CVensembler = crossval(Ensemble, 'KFold', kfold);
%err=kfoldLoss(CVensembler);


test0Results=predict(Ensemble,x2fx(test0Features, 'interaction'));
test0Output=[test0(:,1),test0Results];


%% model for train 1

%kfold=5;

Ensemble1 = fitensemble(x2fx(train1Features, 'quadratic'), train1Revenue,'Bag', 600, 'Tree', 'Type', 'Regression','FResample', 1/5);
%CVensembler1 = crossval(Ensemble1, 'KFold', kfold);
%err1=kfoldLoss(CVensembler1);

test1Results=predict(Ensemble1,x2fx(test1Features, 'quadratic'));
test1Output=[test1(:,1),test1Results];
