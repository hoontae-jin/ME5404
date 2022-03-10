clc; clear; close all;
%% Assign classes
% Studnet matric number : A0243155L
omitted_class1 = mod(5,5); % 0
omitted_class2 = mod(6,5); % 1
% Classes to train: 2, 3 and 4
%% Data Preprocessing (Question 3-c1)
load('Digits.mat');
% training set
trainIdx = find(train_classlabel==2 | train_classlabel==3 | train_classlabel==4);
trainData = train_data(:,trainIdx);
trainLabel= train_classlabel(:,trainIdx);

% test set
testIdx = find(test_classlabel==2 | test_classlabel==3 | test_classlabel==4);
testData = test_data(:,testIdx);
testLabel= test_classlabel(:,testIdx);
%% Given Parameters and details
M = 10; % output row
N = 10; % output column
neurons = M*N; % Number of neurons

iter = 1000; % Number of iterations
lr_0 = 0.1; % Initial Learning rate
sigma_0 = sqrt(M^2+N^2)/2; % Initial effective width
tau_1 = iter/log(sigma_0); % Time constant

% 1. Randomly Initialize the weight vector for neuron i (i=1,...m)
w = rand(size(trainData,1),neurons); % Initial weights
SOM_classlabel = zeros(M,N); % Initial labels
%% Update weights
for i = 0:iter
    lr = lr_0 * exp(-i/iter); % Updated learning rate of "i"th iteration
    sigma = sigma_0 * exp(-i /tau_1); % Updated width of "i"th iteration
    % 2. Sampling: choose an input vector x from the training set.
    for j = 1:size(trainData,2)
        sample = trainData(:,j);
        % 3. Determine winner neuron k
        for k = 1:neurons
            dist(1,k) = norm(w(:,k)-sample);
        end
        [~,winner] = min(dist);
        
        % Obtain indexes of 8x8 matrix for each neuron (winner)
        grid_col = mod(winner,M);
        if grid_col == 0
            grid_col = 10;
        end
        grid_row = ceil(winner/N);
        SOM_classlabel(grid_row,grid_col) = trainLabel(:,j);
        for t = 1:N
            for p = 1:M
                d(t,p) = norm([t,p]-[grid_row,grid_col]); % Euclidean distance
                h(t,p) = exp(-(d(t,p))^2/(2*sigma^2)); % Neighborhood function
            end
        end    
        % Reshape the function to update weight vectors
        h_reshaped = reshape(h',[1,neurons]); 
        % 4. Update all weight vectors of all neurons
        for z = 1:neurons
            w(:,z) = w(:,z) + lr * h_reshaped(1,z) * (sample - w(:,z));
        end
    end
end
%% Plot SOM
SOM_label_flat = reshape(SOM_classlabel,[1 neurons]);
for i = 1:neurons
    subplot(10,10,i)
    w_img = reshape(3*w(:,i),[28 28]); % To make the images more clear, muliply it by 3
    imshow(w_img')
    title(sprintf('%0d',SOM_label_flat(1,i)))
end
%% Question 3-c2
test_SOM_label =[]; % Initial test labels
% 1. Input a test image to SOM, and find out the winner neuron
[row , col] = size(testData);
for i = 1:col
    minimum_dist = 10^8; % initial set-distance to be updated to find the real minimum distance
    for j = 1:neurons
        % 2. label the test image with the winner neuron’s label
        test_dist = norm(testData(:,i)-w(:,j));
        if test_dist < minimum_dist
            minimum_dist = test_dist;
            label_idx = j;
        end
    end
    test_label(1,i) = SOM_label_flat(1,label_idx); %SOM-test label
end
%% plot
for i = 1:col
    subplot(10,10,i)
    test_img = reshape(3*testData(:,i),[28 28]); % To make the images more clear, muliply it by 3
    imshow(test_img')
    title(sprintf('%0d',test_label(1,i))) % around 15 out 60 are wrong (~75%)
end
%% accuracy
correct_label_test = 0;
for i = 1:col
    if test_label(1,i) == testLabel(1,i)
        correct_label_test = correct_label_test + 1;
    end
end
accuracy_test = correct_label_test/col
