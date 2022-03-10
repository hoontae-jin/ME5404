clc; clear; close all
%% Self Organising Maps
N = 2;          % dimnension of input vector
M = 40;         % number of output neurons
iter = 500;     % number of iterations
lr_0 = 0.1;
sigma_0 = sqrt(M^2+N^2)/2;
tau_1 = iter/log(sigma_0);
x = linspace(-pi,pi,400);
trainX = [x; sinc(x)]; % 2x400 matrix

% 1. Randomly Initialize the weight vector for neuron i (i=1,...m)
w = randn(N,M);
for i = 0:iter
    if mod(i,100) == 0
        figure
        plot(trainX(1,:),trainX(2,:),'+r'); axis equal
        hold on
        plot(w(1,:),w(2,:),'-o')
        title(sprintf('SOM (Iteration %d)',i))
        xlabel('x'); ylabel('y')
        legend('y = sinc(x)','SOM');
        grid on;
    end
    lr = lr_0 * exp(-i/iter);
    sigma = sigma_0 * exp(-i /tau_1);
    % 2. Sampling: choose an input vector x from the training set.
    for j = 1:size(trainX,2)
        
        sampling = trainX(:,j);
        
        % 3. Determine winner neuron 
        for k = 1:M
            dist(1,k) = norm(w(:,k)-sampling);
        end
        
        [~,winner] = min(dist);
        % 4. Update all weight vectors of all neurons
        for t = 1:M
            d = t-winner;
            h(1,t) = exp(-d^2 / (2*sigma^2));
            w(:,t) = w(:,t) + lr * h(1,t) * (sampling - w(:,t));
        end
    end
end
%% Sinc Function
function y = sinc(x)
% normalized sinc function, sin(pi*x)/(pi*x), no checks on the input
%
% y = sincc(x)
y = sin(pi*x)./(pi*x);
y(x==0) = 1;
end