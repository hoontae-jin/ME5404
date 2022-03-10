clc; clear; close all;
%% Select two classes
% A0243155L -> Chosen classes : 1 and 5
%% Training without regularisation
load('mnist_m.mat');
% Training
trainIdx = find(train_classlabel==1 | train_classlabel==5);
TrLabel = zeros(size(train_classlabel)); 
for i = 1:length(train_classlabel)
    if train_classlabel(i) == 1 || train_classlabel(i) == 5
        TrLabel(i) = 1;
    end
end
r = exp(-(dist(train_data)).^2/(0.2*10000));
weights = inv(r)*TrLabel';

% Test
testIdx = find(test_classlabel==1 | test_classlabel==5); 
TeLabel = zeros(size(test_classlabel)); 
for i = 1:length(test_classlabel)
    if test_classlabel(i) == 1 || test_classlabel(i) == 5
        TeLabel(i) = 1;
    end
end

r2 = exp(-(dist(test_data',train_data)).^2/(0.2*10000));
y_RBF = r2*weights;

% Please use the following code to evaluate:

TrAcc = zeros(1,1000);
TeAcc = zeros(1,1000);
thr = zeros(1,1000);
TrN = length(TrLabel);
TeN = length(TeLabel);
TrPred = r*weights;
TePred = r2*weights;

for i = 1:1000
    t = (max(TrPred)-min(TrPred)) * (i-1)/1000 + min(TrPred);
    thr(i) = t;
    TrAcc(i) = (sum(TrLabel(TrPred<t)==0) + sum(TrLabel(TrPred>=t)==1)) / TrN;
    TeAcc(i) = (sum(TeLabel(TePred<t)==0) + sum(TeLabel(TePred>=t)==1)) / TeN;
end

figure
plot(thr,TrAcc,'.- ',thr,TeAcc,'^-');
legend('tr','te');
xlabel('Threshold');
ylabel('Accuracy');
title("Accuracy against Threshold without Regularization")