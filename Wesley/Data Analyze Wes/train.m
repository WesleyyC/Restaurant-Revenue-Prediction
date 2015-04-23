%% Trial 1

clear

%% Load Data

train_data = csvread('str_num_train.csv',1,0);
test_data = csvread('str_num_test.csv',1,0);

train_p1p37=train_data(:,6:end-1);
test_p1p37=test_data(:,6:end);


%%
combine=[train_p1p37',test_p1p37']';

