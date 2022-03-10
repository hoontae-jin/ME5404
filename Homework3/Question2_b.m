clc; clear; close all;
%% Obtain Binary TrLabel and TeLabel
load ('mnist_m.mat') 
center_num = 100;
trainidx = find(train_classlabel == 1 | train_classlabel == 5);
TrLabel = zeros(size(train_classlabel)); 
for i = 1:length(train_classlabel)
    if train_classlabel(i) == 1 || train_classlabel(i) == 5
        TrLabel(i) = 1;
    end
end

testidx = find(test_classlabel == 1 | test_classlabel == 5);
TeLabel = zeros(size(test_classlabel)); 
for i = 1:length(test_classlabel)
    if test_classlabel(i) == 1 || test_classlabel(i) == 5
        TeLabel(i) = 1;
    end
end
%% Computation of matrix and weights
shuffle = randperm(size(train_data,2));
idx = shuffle(1,1:center_num);
mu = train_data(:,idx);

for i = 1:center_num
    dis(1,i) = norm(mu(:,i));
end
sigma1 = (max(dis)-min(dis)) /  (sqrt(2*center_num));

%% Interpolation performance and plot graphs
close all
counter = 1;
for sigma = [sigma1, 0.1, 1, 10, 100, 1000, 5000,10000]
    RBF_Training = RBF_func(train_data, sigma,mu);
    RBF_Test = RBF_func(test_data, sigma,mu);
    
    weights = pinv(RBF_Training) * double(TrLabel)';
    
    TrPred = RBF_Training * weights;
    TePred = RBF_Test * weights;
    
    TrAcc = zeros(1,1000);
    TeAcc = zeros(1,1000);
    thr = zeros(1,1000);
    TrN = length(TrLabel);
    TeN = length(TeLabel);
    
    for i = 1:1000
        t = (max(TrPred)-min(TrPred)) * (i-1)/1000 + min(TrPred);
        thr(i) = t;
        TrAcc(i) = (sum(TrLabel(TrPred<t)==0) + sum(TrLabel(TrPred>=t)==1)) / TrN;
        TeAcc(i) = (sum(TeLabel(TePred<t)==0) + sum(TeLabel(TePred>=t)==1)) / TeN;
    end
    
    acc_th(1,counter) = sigma;                      % sigma value
    
    [acc_th(2,counter),thres] = max(TrAcc);         % max training accuracy
    acc_th(3,counter) = thr(1,thres);         
    
    [acc_th(4,counter),thres] = max(TeAcc);         % max testing accuracy
    acc_th(5,counter) = thr(1,thres);         
    
    counter = counter + 1;
    figure;
    plot(thr,TrAcc,'.- ',thr,TeAcc,'^-');legend('tr','te');
    grid
    title(strcat('Accuracy with respect to Threshold (Width = ', " ", num2str(sigma), ")"))
    ylabel("Accuracy"); xlabel("Threshold");
end
%% Accuracy with respect to width
figure;
hold on
plot(acc_th(1,:),acc_th(2,:),'-m');
plot(acc_th(1,:),acc_th(4,:),'-k');
legend('Training data','Test data','Location','northeast');
grid
title('Accuracy with respect to differnt widths');
ylabel("Accuracy"); xlabel("Width");