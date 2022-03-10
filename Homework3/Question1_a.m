clc; clear all; close all;
%% Training
x_training = -1.6:0.08:1.6;
y_training = 1.2*sin(pi*x_training) - cos(2.4*pi*x_training);
noise = randn(size(x_training));
True_output = 1.2*sin(pi*x_training) - cos(2.4*pi*x_training) + 0.3*noise;
r = exp(-(dist(x_training)).^2/(2*0.01)); % Gaussian function with Euclidean distance
weights = inv(r)*True_output';
%% Test
x_test = -1.6:0.01:1.6;
y_test = 1.2*sin(pi*x_test) - cos(2.4*pi*x_test);
r2 = exp(-(dist(x_test',x_training)).^2/(2*0.01));
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
legend('Training Sample','Desired output','RBFN output')
txt = ["Training MSE","MSE1"];
hold off
%% MSE 
syms x
weights;
for i = 1:length(x_training)
    r_syms(i,1) = exp( - (x-x_training(1,i))^2 / (2*0.01));
end
y_RBF_syms = dot(weights,r_syms);
y_train_result = subs(y_RBF_syms,x,x_training);
MSE_training = eval(sum((y_train_result - True_output).^2)/length(x_training))
MSE_test = sum((y_RBF - y_test).^2)/length(x_test)