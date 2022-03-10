clc; clear; close all
%% Given parameters
M = 8; % Horizontal Neurons
N = 8; % Vertical Neurons
neurons = M*N; % number of Neurons

iter = 500; % Number of iterations
lr_0 = 0.1; % Initial Learning rate
sigma_0 = sqrt(M^2+N^2)/2; % Initial effective width
tau_1 = iter/log(sigma_0); % Time constant

% 1. Randomly Initialize the weight vector for neuron i (i=1,...m)
w = rand(2,neurons); % Initial weights
%% Training set
X = randn(800,2); % Training
s2 = sum(X.^2,2); % Training 
trainX = (X.*repmat(1*(gammainc(s2/2,1).^(1/2))./sqrt(s2),1,2))';% Training
%% Initial Plot (Totally un-organised)
plot(trainX(1,:),trainX(2,:),'+r'); axis equal
hold on
plot(w(1,:),w(2,:),'-b'); axis equal
%% Update weights
for i = 0:iter
    lr = lr_0 * exp(-i/iter); % Updated learning rate of "i"th iteration
    sigma = sigma_0 * exp(-i /tau_1); % Updated width of "i"th iteration
    % 2. Sampling: choose an input vector x from the training set.
    for j = 1:length(trainX)
        sample = trainX(:,j);
        % 3. Determine winner neuron k
        for k = 1:length(w)
            dist(:,k) = norm(w(:,k)-sample);
        end
        [~,winner] = min(dist);
        
        % Obtain indexes of 8x8 matrix for each neuron (winner)
        grid_col = mod(winner,M);
        if grid_col == 0
            grid_col = 8;
        end
        grid_row = ceil(winner/N);
        
        for t = 1:N
            for p = 1:M
                d(t,p) = norm([t,p]-[grid_row,grid_col]); % Euclidean distance
                h(t,p) = exp(-(d(t,p))^2/(2*sigma^2)); % Neighborhood function
            end
        end
        
        % Reshape the function to update weight vectors
        h_reshaped = reshape(h',[1,64]); 
        % 4. Update all weight vectors of all neurons
        for z = 1:neurons
            w(:,z) = w(:,z) + lr * h_reshaped(1,z) * (sample - w(:,z));
        end
    end
    
    w_reshaped = reshape(w',[8 8 2]);
    if logical(mod(i,100)) == 0
        plot(trainX(1,:),trainX(2,:),'+r')
        hold on
        plot(w_reshaped(:,:,1), w_reshaped(:,:,2), '+b')
        for Z = 1:8
            plot(w_reshaped(:,Z,1),w_reshaped(:,Z,2),'k-')
            plot(w_reshaped(Z,:,1),w_reshaped(Z,:,2),'k-')
        end
        xlabel('x'); ylabel('y')
        title(sprintf('SOM (Iteration %d)',i))
        hold off
    end
end
        
        
        