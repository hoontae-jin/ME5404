clc; clear; close all;
%% Load the discriminant variables
load('discriminant_variables_linear.mat')
load('discriminant_variables_poly.mat')
load('discriminant_variables_tanh.mat')
%% Standardization of the test dataset
load('test.mat')
[m, n] = size(test_data);
meanFeature = mean(test_data,2); 
stdFeature = std(test_data,0,2); 
new_d = test_data - repmat(meanFeature,1,n);
strd_Testdata = new_d./stdFeature;
p=5; % Change this value accordingly with the table given 
%% Accuracy check of the training dataset
% Hard-margin with Linear Kernel
disp(['--------------Accuracy for Training dataset-----------------------'])
kernel = 'linear';
[accuracy_linear] = acc_check(strd_data, train_label, strd_data, train_label, w_linear, b_linear, a_linear, kernel, 0);
disp(['Hard-Margin with the linear kernel :', num2str(accuracy_linear),'%'])

% Hard-margin with Polynomial Kernel
kernel = 'poly';
[accuracy_poly] = acc_check(strd_data, train_label, strd_data, train_label, w_poly, b_poly, a_poly, kernel, p);
disp(['Hard-Margin with the polynomial kernel :', num2str(accuracy_poly),'%'])

% Soft-margin with Polynomials
kernel = 'tanh';
[accuracy_tanh] = acc_check(strd_data, train_label, strd_data, train_label, w_tanh, b_tanh, a_tanh, kernel, p);
disp(['Soft-Margin(tanh) with the polynomial :', num2str(accuracy_tanh),'%'])
disp(['------------------------------------------------------------------'])
%% Accruacy check of the test dataset
% Hard-margin with Linear Kernel
disp(['---------------------Accuracy for Test dataset--------------------'])
kernel = 'linear';
[accuracy_linear] = acc_check(strd_Testdata, test_label, strd_data, train_label, w_linear, b_linear, a_linear, kernel, 0);
disp(['Hard-Margin with the linear kernel :', num2str(accuracy_linear),'%'])

% Hard-margin with Polynomial Kernel
kernel = 'poly';
[accuracy_poly] = acc_check(strd_Testdata, test_label, strd_data, train_label, w_poly, b_poly, a_poly, kernel, p);
disp(['Hard-Margin with the polynomial kernel :', num2str(accuracy_poly),'%'])

% Soft-margin with Polynomials
kernel = 'tanh';
[accuracy_tanh] = acc_check(strd_Testdata, test_label, strd_data, train_label, w_tanh, b_tanh, a_tanh, kernel, p);
disp(['Soft-Margin(tanh) with the polynomial :', num2str(accuracy_tanh),'%'])
disp(['------------------------------------------------------------------'])
%% Function for Accuracy
function [accuracy] = acc_check(testData, testLabel, trainData, trainLabel, w, b, a, kernel, p)
switch kernel
    case 'linear'
        [~, n] = size(testData); %n=1536
        predicted_label = zeros(n,1);%1536x1
        accuracy_num = zeros(n,1); %1536x1
        for i = 1:n
            predicted_label(i,1) = (w'*testData(:,i) + b);
            if predicted_label(i,1) > 0
               predicted_label(i,1) = 1;  
            elseif predicted_label(i,1) <=0
               predicted_label(i,1) = -1; 
            end
        end
        for i = 1:n
            if predicted_label(i,1) == testLabel(i,1)
                accuracy_num(i,1) = 1;
            elseif predicted_label(i,1) ~= testLabel(i,1)
                accuracy_num(i,1) = 0;
            end
        end
        accuracy = (sum(accuracy_num)/n)*100;
        
        
    case 'poly'
        M = length(testLabel);
        n = length(trainLabel);
        predicted_label = zeros(M,1);
        gx = zeros(M,1); 
        for j = 1:M        
            wx = 0;
            for i = 1:n
                wx = wx + a(i,:) * trainLabel(i,:) * (testData(:,j)' * trainData(:,i) + 1) ^ p;
            end
            gx(j) = wx + b;
        end
        for i = 1:M
            if gx(i) > 0
               predicted_label(i,1) = 1;  
            else
               predicted_label(i,1) = -1; 
            end
        end
        accuracy = sum(predicted_label==testLabel) / M * 100;
        
    case 'tanh'
        M = length(testLabel);
        n = length(trainLabel);
        predicted_label = zeros(M,1);
        gx = zeros(M,1); 
        for j = 1:M        
            wx = 0;
            for i = 1:n
                wx = wx + a(i,:) * trainLabel(i,:) * tanh(testData(:,j)' * trainData(:,i) -1) ^ p;
            end
            gx(j) = wx + b;
        end
        for i = 1:M
            if gx(i) > 0
               predicted_label(i,1) = 1;  
            else
               predicted_label(i,1) = -1; 
            end
        end
        accuracy = sum(predicted_label==testLabel) / M * 100;        
end
end

