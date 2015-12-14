function [training, trainingClass, validation, validationClass, test, testClass] = ourAdaptedSmote(data, class, k)

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

%Applying smote function to unbalanced class training and validation set
newTraining1= training1;
newValidation1= validation1;

while size(newTraining1,1)<size(training0,1)
    tempT=smoteAdaptAlgorithm([training0; training1], [zeros(size(training0, 1), 1); ones(size(training1, 1), 1)], training1, k);
    newTraining1=[newTraining1;tempT];
    tempV=smoteAdaptAlgorithm([validation0; validation1], [zeros(size(validation0, 1), 1); ones(size(validation1, 1), 1)], validation1, k);
    newValidation1=[newValidation1;tempV];
end

training1=newTraining1(1:size(training0,1),:);
validation1=newValidation1(1:size(validation0,1),:);

%Merging training, validation and test sets in their respective matrices
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

function smoteData = smoteAdaptAlgorithm(data, class, minorityData, k)

% Preallocate for performance
smoteData = zeros(k * size(minorityData, 1), size(minorityData, 2));

% Since the nearest neighbor is always the own sample, we search for its
% k+1 closest neighbors and discard the first one
IDX = knnsearch(data, minorityData, 'k', (k + 1));
IDX = IDX(:, 2:end);

% Generate the synthetic samples
for i = 1:size(IDX, 1)
    for j = 1:size(IDX, 2)
        if class(IDX(i, j), 1) == 1 %It belongs to the minor class 
            smoteData(((i-1) * k) + j, :) = minorityData(i, :) + (rand * (data(IDX(i, j), :) - minorityData(i, :)));
        else  %It belongs to the major class
            smoteData(((i-1) * k) + j, :) = minorityData(i, :) + (rand/2 * (data(IDX(i, j), :) - minorityData(i, :)));
        end
    end
end
end




