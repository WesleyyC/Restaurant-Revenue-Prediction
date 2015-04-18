

result = csvread('submit_boost.csv',1,1);


output=result+randn(size(result))*5e3;

output=int64(output);

id=0:length(output)-1;

output=[id',output];
