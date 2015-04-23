output1=csvread('1.csv',1,1);
output2=csvread('2.csv',1,1);
output3=csvread('3.csv',1,1);

output=output1*1+output2*3+output3*4;
output=output/8;