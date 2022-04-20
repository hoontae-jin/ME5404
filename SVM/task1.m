clc; clear; close all;
%% Data Normalization for the training set
load('train.mat')
[m, n] = size(train_data);
meanFeature = mean(train_data,2); 
stdFeature = std(train_data,0,2); 
new_d = train_data - repmat(meanFeature,1,n);
strd_data = new_d./stdFeature; 

%% Hard-margin SVM with the linear kernel
%------------------------Important notes-----------------------------------
% Maximizing: Q(a) = sigma(a(i)) - (1/2)*sigma(a(i)a(j)d(i)d(j)(x(i)^T)x(j)
% Subject to: sigma(a(i)d(i))=0 and a(i) >= 0
% where a(i and j) : The only unknowns (Lagrange Multipliers)
%       d(i and j) : The true outputs
%       x(i and j) : The inputs
%--------------------------------------------------------------------------
% Step1 : Calculate Gram matrix (K) and check the Mercer condition
% K(x1,x2) = transpose(x1)*x2
K_linear = zeros(m,n);
% Q_partial is the part of Q(a) ~ {d(i)d(j)(x(i)^T)x(j)} where  K(i,j) = (x(i)^T)x(j)
Q_linear = zeros(m,n);
for i = 1:n
    for j = 1:i
        % K(x1,x2) = transpose(x1)*x2
        K_linear(i,j) = transpose(strd_data(:,i))*strd_data(:,j);
        % Compute the part of Q(a)
        Q_linear(i,j) = train_label(i)*train_label(j)*K_linear(i,j);
        % Construct the symmetric matrices for K and Q_partial. This is a
        % required step to proceed with the optimization method
        K_linear(j,i) = K_linear(i,j);
        Q_linear(j,i) = Q_linear(i,j);
    end
end
threshold = 10^-4;
Check_mercer(K_linear, threshold)
% Step2 : Calculate the Lagrange Multipliers (a)
%--------------------------important notes---------------------------------
% Use "quadprog" to compute the required constraint
% min((1/2)*x'*H*x + f'*x
% Conditions: A*b =< b, Aeq*x = beq, lb =< x =< ub)
% where H, A and Aeq are matrices, and f,b,beq,lb,ub and x are vectors
% H : Q_partial
% Ae: Transpose(train_label)
%--------------------------------------------------------------------------
[m, ~] = size(K_linear);
C = 100; % When C=10^6 (given in the demo pdf), the computation cost is too expensive.
f = -ones(m,1); % 
% Let the constraint values in the range (0 and C)
lb = zeros(m,1); ub = ones(m,1) * C;
%options = optimset('LargeScale','off','MaxIter',1000);
options = optimset('LargeScale','off','MaxIter',1000);
a_linear = quadprog(Q_linear,f,[],[],transpose(train_label),0,lb,ub,[],options);
for i = 1 : size(a_linear)
    threshold = 10^-4; % User-defined value
    if a_linear(i,1) < threshold
        a_linear(i,1) = 0;
    end
end
svm_idx_linear = find(a_linear>0 & a_linear<C);
% Step3 : Compute the optimal weights and bias
% The equation of the weight update for the linear kernel is :
% w = sig(a(i)*d(i)*x(i)
% The equation of the bias for the linear kernel is :
% b = (1/d(s)) - w'*x(s) where s is a support vector with label d(s)
[~, n] = size(strd_data);
w_linear = 0; % weight
for i = 1:n
    w_linear = w_linear + a_linear(i,1)*train_label(i,1)*strd_data(:,i);
end
b_linear = mean((1 ./train_label(svm_idx_linear)')-(transpose(w_linear)*strd_data(:,svm_idx_linear))); % bias
% Step 4: Save discriminant_variables
save('discriminant_variables_linear');
%% Other Kernels (Hard and Soft-Margin SVM with polynomial)
% The steps for these kernels are similar to the above-procedures except
% for the equation differences. Hence, functions were created to simplify
% the script. The functions can be found at the bottom
%% Hard-Margin SVM with polynomial
clc; clear; close all;
load('train.mat')
[m, n] = size(train_data);
meanFeature = mean(train_data,2); 
stdFeature = std(train_data,0,2); 
new_d = train_data - repmat(meanFeature,1,n);
strd_data = new_d./stdFeature; 

kernel = 'poly'; % polynomial and tanh.
p = 5;
C = 100;
threshold = 10^-4;
% Step 1: Calculate Gram matrix (K) and check the Mercer condition
[K_poly, Q_poly] = GramMat(strd_data,train_label,kernel,p);
Check_mercer(K_poly, threshold)
% Step 2: Calculate the Lagrange Multipliers (a)
[a_poly, svm_idx_poly] = optim_quadprog(Q_poly, train_label,C,threshold);
% Step 3: Calculate weights and bias
[w_poly, b_poly] = wb_compute(strd_data,train_label,svm_idx_poly,kernel,p,a_poly);
% Step 4: Save discriminant_variables
save('discriminant_variables_poly');
%% Soft-Margin SVM with polynomial
clc; clear; close all;
load('train.mat')
[m, n] = size(train_data);
meanFeature = mean(train_data,2); 
stdFeature = std(train_data,0,2); 
new_d = train_data - repmat(meanFeature,1,n);
strd_data = new_d./stdFeature; 

kernel = 'tanh';
p = 1;
C = 1.1; % 0.1 and 0.6 are too low to produce weight and bias
threshold = 10^-4;
% Step 1: Calculate Gram matrix (K) and check the Mercer condition
[K_tanh, Q_tanh] = GramMat(strd_data,train_label,kernel,p);
Check_mercer(K_tanh, threshold)
% Step 2 : Calculate the Lagrange Multipliers (a)
[a_tanh, svm_idx_tanh] = optim_quadprog(Q_tanh, train_label, C, threshold);
% Step 3: Calculate weights and bias
[w_tanh, b_tanh] = wb_compute(strd_data,train_label,svm_idx_tanh,kernel,p,a_tanh);
% Step 4: Save discriminant_variables
save('discriminant_variables_tanh');
%% Run this part to obtain the accuracy for the training and the test datasets
% For the polynomial kernels, change its value manually in both files
% (task1.m & task2.m)
run task2.m
%% Functions for step1, 2 and 3
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
    case 'poly'
        for i = 1:n
            for j = 1:i
                K(i,j) = ((trainData(:,i))'*trainData(:,j)+1)^p;
                Q_partial(i,j) = trainLabel(i)*trainLabel(j)*K(i,j);
                K(j,i) = K(i,j);
                Q_partial(j,i) = Q_partial(i,j);
            end
        end
    case 'tanh'      
        for i = 1:n
            for j = 1:i
                K(i,j) = tanh((trainData(:,i))'*trainData(:,j)-1)^p;
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
        
    case 'poly'
        n = length(trainLabel);
        b = zeros(size(svm_idx));
        for i = 1:length(svm_idx)
            idx = svm_idx(i);
            w = 0;
            for j = 1:n
                w = w + a(j,:)*trainLabel(j,:)*(trainData(:,idx)'*trainData(:,j)+1)^p;
            end
            b(i) = trainLabel(idx,:) - w;
        end
        b0 = mean(b);
                     
    case 'tanh'
        n = length(trainLabel);
        b = zeros(size(svm_idx));
        for i = 1:size(svm_idx,1)
            idx = svm_idx(i);
            w = 0;
            for j = 1:n
                w = w + a(j,:)*trainLabel(j,:)*(tanh( (trainData(:,idx)'*trainData(:,j))-1)^p);
            end
            b(i) = trainLabel(idx,:) - w;
        end
        b0 = mean(b);
end
end

