%Read file: Mamography 
originalData= csvread('mammography-consolidated.csv');

%Put the class atribute into another matrix
class = originalData(:, end);
data= originalData(:, 1:(end-1));

%Number of neighbours chosen when applying smote algorithm
k=5;

%Divide the dataset in training 50%, validation 25%, test 25%
%[training,trainingClass, validation, validationClass, test, testClass]=oversample(data, class);
%[training,trainingClass, validation, validationClass, test, testClass]=ourSmote(data, class,k);
%[training,trainingClass, validation, validationClass, test, testClass]=smote(data, class,k);
%[training,trainingClass, validation, validationClass, test, testClass]=adaptedSmote(data, class,k);
[training,trainingClass, validation, validationClass, test, testClass]= undersample(data, class);


%Write balanced data into a csv file
csvwrite('mammography-consolidated-training-undersample.csv',[training, trainingClass])
csvwrite('mammography-consolidated-validation-undersample.csv',[validation, validationClass])
csvwrite('mammography-consolidated-test-undersample.csv',[test, testClass])