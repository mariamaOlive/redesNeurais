%Read file: Mamography 
originalData= csvread(mammography-consolidated.csv);


%Divide the dataset in training 50%, validation 25%, test 25%
[training, validation, test]=oversample(data);