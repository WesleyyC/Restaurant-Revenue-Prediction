

result = csvread('kill wol.csv',1,1);


output=result+randn(size(result))*100;

output=int64(output);

id=0:length(output)-1;

output=[id',output];
