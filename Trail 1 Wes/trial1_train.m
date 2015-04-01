%% Trial 1

clear

%% Load Data

raw_data = csvread('str_num_train.csv',1,0,100000,30);

revenue=raw_data(43)