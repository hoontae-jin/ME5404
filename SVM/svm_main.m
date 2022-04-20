clc; clear; close all;
%% Create Evaluation Datasets using the training and test datasets
% This shows how the eval dataset was created. Do not run this code as it
% can result in different accuracy value due to the random shuffling for
% each run. The original eval.mat is already located in the folder.
%load('train.mat'); load('test.mat');

%data = [train_data, test_data];
%label = [train_label ; test_label];
%combined = [data ; label'];

%cols = size(combined,2);
%P = randperm(cols);
%shuffle = combined(:,P);

%eval_data = shuffle(1:57,1:600);
%eval_label = shuffle(58,1:600)';
%save('eval.mat','eval_data','eval_label')
%% 1. Obtain a new discriminant function for the RBF kernel
load('train.mat')
[m, n] = size(train_data);
meanFeature = mean(train_data,2); 
stdFeature = std(train_data,0,2); 
new_d = train_data - repmat(meanFeature,1,n);
strd_data = new_d./stdFeature; 

kernel = 'rbf';
p = 5; % Sigma
C = 2.1;
threshold = 10^-4;
% Step 1: Calculate Gram matrix (K) and check the Mercer condition
[K_rbf, Q_rbf] = GramMat(strd_data,train_label,kernel,p);
Check_mercer(K_rbf, threshold)
% Step 2 : Calculate the Lagrange Multipliers (a)
[a_rbf, svm_idx_rbf] = optim_quadprog(Q_rbf, train_label, C, threshold);
% Step 3: Calculate weights and bias
[w_rbf, b_rbf] = wb_compute(strd_data,train_label,svm_idx_rbf,kernel,p,a_rbf);
% Step 4: Save discriminant_variables
save('discriminant_variables_rbf');
%% 2. Discriminant functions of Linear kernel and RBF kernel
load('discriminant_variables_linear.mat')
load('discriminant_variables_rbf.mat')
%% 3. Standardization for the evaluation dataset
load('eval.mat')
[m, n] = size(eval_data);
meanFeature = mean(eval_data,2); 
stdFeature = std(eval_data,0,2); 
new_d = eval_data - repmat(meanFeature,1,n);
strd_Evaldata = new_d./stdFeature;
%% Accuracy Check
disp(['--------------Accuracy for Training dataset-----------------------'])
kernel = 'linear';
[accuracy_linear, training_predicted_linear] = acc_check(strd_data, train_label, strd_data, train_label, w_linear, b_linear, a_linear, kernel, p);
disp(['Hard-Margin with the linear kernel :', num2str(accuracy_linear),'%'])

kernel = 'rbf';
[accuracy_rbf, training_predicted_rbf] = acc_check(strd_data, train_label, strd_data, train_label, w_rbf, b_rbf, a_rbf, kernel, p);
disp(['Soft-Margin with the RBF kernel :', num2str(accuracy_rbf),'%'])

disp(['---------------------Accuracy for Eval dataset--------------------'])
kernel = 'linear';
[accuracy_linear, eval_predicted_linear] = acc_check(strd_Evaldata, eval_label, strd_data, train_label, w_linear, b_linear, a_linear, kernel, p);
disp(['Hard-Margin with the linear kernel :', num2str(accuracy_linear),'%'])

kernel = 'rbf';
[accuracy_rbf, eval_predicted_rbf] = acc_check(strd_Evaldata, eval_label, strd_data, train_label, w_rbf, b_rbf, a_rbf, kernel, p);
disp(['Soft-Margin with the RBF kernel :', num2str(accuracy_rbf),'%'])
%% Predicted eval labels
eval_predicted = eval_predicted_rbf;
%% Functions (Hard margin with the polynomial kernel vs RBF
function [K, Q_partial] = GramMat(trainData,trainLabel,kernel,p)
% step 1.  Calculate Gram matrix (K)
[m, n] = size(trainData);
K = zeros(m,n);
Q_partial = zeros(m,n);
switch kernel          
    case 'linear'
        for i = 1:n
            for j = 1:i
                K(i,j) = transpose(trainData(:,i))*trainData(:,j);
                Q_partial(i,j) = trainLabel(i)*trainLabel(j)*K(i,j);
                K(j,i) = K(i,j);
                Q_partial(j,i) = Q_partial(i,j);
            end
        end  
    case 'rbf'      
        for i = 1:n
            for j = 1:i
                K(i,j) = exp(-norm(trainData(:,i)-trainData(:,j))/(p^2));
                Q_partial(i,j) = trainLabel(i)*trainLabel(j)*K(i,j);
                K(j,i) = K(i,j);
                Q_partial(j,i) = Q_partial(i,j);
            end
        end
end
end
function Check_mercer(K, threshold)
eigen_vals = eig(K);
eigen_check = eigen_vals < 0;
min_e = abs(min(eigen_vals(eigen_vals < 0)));

if (sum(eigen_check) == 0) || (threshold > min_e)
    disp('Satisfies Mercer condition');
else
    disp('Does not satisfy Mercer condition');

end
end
function [a, svm_idx] = optim_quadprog(Q_partial, trainLabel, C, threshold)
m = length(Q_partial);
f = -ones(m,1); 
A = [];
b = [];
Aeq = trainLabel'; beq = 0;
lb = zeros(m,1); ub = ones(m,1) * C;
x0 = []; options = optimset('LargeScale','off','MaxIter',1000);
a = quadprog(Q_partial,f,A,b,Aeq,beq,lb,ub,x0,options);

for i = 1 : size(a)
    if a(i,1) < threshold
        a(i,1) = 0;
    end
end
svm_idx = find(a>0 & a<C);
end
function [w, b0] = wb_compute(trainData,trainLabel,svm_idx,kernel,p,a)
switch kernel
    case 'linear'
        n = length(trainLabel);
        w = 0;
        for i = 1:n
            w = w + a(i,1)*trainLabel(i,1)*trainData(:,i);
        end
        b0 = mean((1 ./trainLabel(svm_idx)') - (transpose(w)*trainData(:,svm_idx)));            
    case 'rbf'
        n = length(trainLabel);
        b = zeros(size(svm_idx));
        for i = 1:size(svm_idx,1)
            idx = svm_idx(i);
            w = 0;
            for j = 1:n
                w = w + a(j,:)*trainLabel(j,:)*(exp(-norm(trainData(:,idx)-trainData(:,j))/(p^2)));
            end
            b(i) = trainLabel(idx,:) - w;
        end
        b0 = mean(b);
end
end
function [accuracy, predicted_label] = acc_check(testData, testLabel, trainData, trainLabel, w, b, a, kernel, p)
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
    case 'rbf'
        M = length(testLabel);
        n = length(trainLabel);
        predicted_label = zeros(M,1);
        gx = zeros(M,1); 
        for j = 1:M        
            wx = 0;
            for i = 1:n
                wx = wx + a(i,:) * trainLabel(i,:) * exp(-norm(testData(:,j) - trainData(:,i))/(p^2));
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