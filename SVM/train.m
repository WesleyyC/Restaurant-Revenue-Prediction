
trainData = csvread('str_num_train.csv',1,0);
trainFeatures = [trainData(:,2),trainData(:,4:end-1)];
trainRevenue = trainData(:, end:end);

testData = csvread('str_num_test.csv',1,0);
testFeatures=[testData(:,2),testData(:,4:end)];
testLabelinput=zeros([100000,1]);
%% Handle Outline

for i = 1:length(trainRevenue)
    
    if trainRevenue(i)>1e7
        trainRevenue(i)=1e7;
    end
    
end


%% train svm

model=svmtrain(trainRevenue,trainFeatures,'-s 4 -t 0');


%% predict result
[output, accuracy, decision_values] = svmpredict(testLabelinput, testFeatures, model);