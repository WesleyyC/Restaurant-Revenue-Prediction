%% Trial 1

clear

%% Load Data

raw_data = csvread('str_num_train.csv',1,0);
test_data=csvread('str_num_test.csv',1,0);

%% Load XY
Y = raw_data(:,43);
X = raw_data(:,3:42);
%X=zscore(X);

X_t = test_data(:,3:42);
%X_t =zscore(X_t);



%% Run model with crossval

modelspec={'constant','linear','interactions','purequadratic','quadratic'};

%for i = 1:5
    mdl=fitlm(X,Y,modelspec{2});
    %val=crossval(mdl,X);
    
%end





%%
% %% Outliner Check
% 
% greater1e7 = Y>1e7;
% 
% N = 30;
% err=ones(N);
% 
% for n = 1:N
%     tree=fitctree(X,greater1e7,'CrossVal','on','KFold',3,'MaxNumCategories',n);
%     err(n)=kfoldLoss(tree);
% end
% 
% 
% 
% %% Tree Result
% class_2=predict(tree,X);
% class_1=1-class_2;
% 
% X_1=X(find(class_1),:);
% Y_1=Y(find(class_1));
% 
% X_2=X(find(class_2),:);
% Y_2=Y(find(class_2));

