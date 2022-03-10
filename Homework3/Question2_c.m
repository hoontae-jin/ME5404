close all; clear; clc;
%% TrLabel and TeLabel Preparation
load ('mnist_m.mat') 
num_cen = 2;
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
%% Kmeans clustering and calculate width
[~, c] = kmeans(train_data',num_cen); % c: center, kmeans: the built-in function in matlab

%% Calculate interpolation matrix and weights
close all
counter = 1;
for sigma = [0.1, 1, 10, 100, 1000, 5000,10000]
    RBF_Training = RBF_func(train_data, sigma,c');
    RBF_Test = RBF_func(test_data, sigma,c');
    
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

figure;
hold on
plot(acc_th(1,:),acc_th(2,:),'-m');
plot(acc_th(1,:),acc_th(4,:),'-k');
legend('Training data','Test data','Location','northeast');
grid
title('Accuracy against Width');
ylabel("Accuracy"); xlabel("Width");

%% Plot centers and mean   
lab0_idx = find(TrLabel ~= 1);
lab1_idx = find(TrLabel ~= 0);

lab1_mean = mean(train_data(:,lab1_idx),2);
lab0_mean = mean(train_data(:,lab0_idx),2);

plot_img(c','Center'); %k-means clustering 
plot_img(lab1_mean,'Label 1');  
plot_img(lab0_mean,'Label 0');     
%% Function
function plot_img(data,lab)
num_data = size(data, 2);
for col_num = 1:num_data
    img = reshape(data(:,col_num),[28 28]);
    figure;
    imshow(img','InitialMagnification',1000 );
    title(lab)
end
end
