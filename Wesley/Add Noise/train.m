

result = csvread('submit_boost.csv',1,1);


output=result-5e4+randn(size(result))*1000;

output=int64(output);

id=0:length(output)-1;

output=[id',output];
