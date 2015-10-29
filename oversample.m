function [training,trainingClass, validation, validationClass, test, testClass] = oversample(data, class)

%Divide the dataset into two classes
a=class==0;
b=class==1;

data0= data(a, :);
data1= data(b, :);

%Randomize data
random=randperm(size(data0,1));
data0=data0(random,:);
random=randperm(size(data1,1));
data1=data1(random,:);

%Divide the class0 in training 50%, validation 25% and test 25%

[numberRows0,~]=size(data0);

nRowsTrn0= fix(0.5*numberRows0);
nRowsVdt0= fix(0.25*numberRows0);

startVld= nRowsTrn0+1;
startTest= nRowsTrn0+nRowsVdt0+1;

training0= data0(1:nRowsTrn0,:);
validation0= data0((startVld):(startTest-1),:);
test0= data0(startTest:end,:);

%Divide the class1 in training 50%, validation 25% and test 25%
%Remember class 1 is smaller than 0 so we have to balance it!

[numberRows1,~]=size(data1);

nRowsTrn1= fix(0.5*numberRows1);
nRowsVdt1= fix(0.25*numberRows1);

startVld= nRowsTrn1+1;
startTest= nRowsTrn1+nRowsVdt1+1;

training1= data1(1:nRowsTrn1,:);
validation1= data1((startVld):(startTest-1),:);
test1= data1(startTest:end,:);



%Balancing class1 in order to be the same size as class0
%Training
[sizeTraining0,~]=size(training0);
[sizeTraining1,~]=size(training1);
nReplications= fix((sizeTraining0-sizeTraining1)/sizeTraining1);
training1=repmat(training1, nReplications,1);
[auxSize,~]=size(training1);
incr=sizeTraining0-auxSize;
training1= [training1; training1((1:incr),:)];

%Validation
[sizeValidation0,~]=size(validation0);
[sizeValidation1,~]=size(validation1);
nReplications= fix((sizeValidation0-sizeValidation1)/sizeValidation1);
validation1=repmat(validation1, nReplications,1);
[auxSize,~]=size(validation1);
incr=sizeValidation0-auxSize;
validation1= [validation1; validation1((1:incr),:)];

%Merging training, validation and test (class0+class1)
%Training
[nRows0, ~]=size (training0);
[nRows1, ~]=size (training1);

trainingClass=[zeros(nRows0,1);ones(nRows1,1)];
training=[training0;training1];
%Shuffling training set
random=randperm(size(training,1));
training= training(random,:);
trainingClass=trainingClass(random,:);

%Validation
[nRows0, ~]=size (validation0);
[nRows1, ~]=size (validation1);

validationClass=[zeros(nRows0,1);ones(nRows1,1)];
validation=[validation0;validation1];
%Shuffling validation set
random=randperm(size(validation,1));
validation= validation(random,:);
validationClass= validationClass(random,:);

%Test
[nRows0, ~]=size (test0);
[nRows1, ~]=size (test1);
testClass=[zeros(nRows0,1);ones(nRows1,1)];
test=[test0; test1];
end




