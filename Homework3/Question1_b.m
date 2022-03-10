clc; clear all; close all;
%% Training
x_training = -1.6:0.08:1.6;
noise = randn(size(x_training));
y_training = 1.2*sin(pi*x_training) - cos(2.4*pi*x_training);
True_output = 1.2*sin(pi*x_training) - cos(2.4*pi*x_training) + 0.3*noise;
shuffle = randperm(length(x_training)); % Shuffle the indexes of x_training
mu = x_training(shuffle(1:20)); % Select 20 centers randomly
d_max = max(max(dist(mu))); % Maximum distance between the chosen centres
r = exp(-(20/(d_max^2))*dist(x_training',mu).^2); % Gaussian function with Euclidean distance
spreads = d_max/sqrt(2*20);
weights = pinv(r)*True_output';
%% Test
x_test = -1.6:0.01:1.6;
y_test = 1.2*sin(pi*x_test) - cos(2.4*pi*x_test);
r2 = exp(-(20/(d_max^2))*dist(x_test',mu).^2);
y_RBF = (r2*weights)';
%% Plot
figure
plot(x_training,True_output,'ks-');
hold on
plot(x_test,y_test,'r-');
hold on;
plot(x_test,y_RBF,'b+');
xlabel('x')
ylabel('y')
legend('Training sample','Desired output','RBFN')
hold off
%% MSE
syms x
weights;
for i = 1:length(mu)
    r_syms(i,1) = exp( -(20/(d_max^2))*(x-mu(1,i))^2);
end
y_RBF_syms = dot(weights,r_syms);
y_train_result = subs(y_RBF_syms,x,x_training);
MSE_training = eval(sum((y_train_result - True_output).^2)/length(x_training))
MSE_test = sum((y_RBF - y_test).^2)/length(x_test)