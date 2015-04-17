


%% cluster city type
trainData = csvread('str_num_test.csv',1,0);
cityResult = csvread('city.csv',1,0);
willResult = csvread('1.csv',1,0);

train0Result = cityResult(~ismember(trainData(:,4),[1]),:);
train1Result = willResult(~ismember(trainData(:,4),[0]),:);

