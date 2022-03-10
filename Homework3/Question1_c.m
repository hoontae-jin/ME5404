clc; clear all; close all;
%% Training
x_training = -1.6:0.08:1.6;
y_training = 1.2*sin(pi*x_training) - cos(2.4*pi*x_training);
noise = randn(size(x_training));
True_output = 1.2*sin(pi*x_training) - cos(2.4*pi*x_training) + 0.3*noise;
r = exp(-(dist(x_training)).^2/(2*0.01)); % Gaussian function with Euclidean distance
%% Test
lamda = [0.001, 0.05, 0.01, 0.5, 0.1, 10];
for i = 1:6
    w_new = inv(r'*r + lamda(i)*eye(size(r)))*r'*True_output';
    x_test = -1.6:0.01:1.6;
    y_test = 1.2*sin(pi*x_test) - cos(2.4*pi*x_test);
    r2 = exp(-(dist(x_test',x_training)).^2/(2*0.01));
    y_RBF = (r2*w_new)';
    MSE_test(1,i) = sum((y_RBF - y_test).^2)/length(x_test);
    
    figure
    plot(x_training,True_output,'ks-');
    hold on
    plot(x_test,y_test,'r-');
    hold on;
    plot(x_test,y_RBF,'b+');
    xlabel('x')
    ylabel('y')
    legend('Training','Test','RBFN')
    title(strcat('(Regularization = ', " ", num2str(lamda(i)), ")"))
end
%% MSE
syms x
for i = 1:length(lamda)
    w_new = inv(r'*r + lamda(i)*eye(size(r)))*r'*True_output';
    for j = 1:length(x_training)   
        r_syms(j,1) = exp( -(x-x_training(1,j))^2 / (2*0.01));
    end  
    y_RBF_syms = dot(w_new,r_syms);
    y_train_result = subs(y_RBF_syms,x,x_training);
    MSE_training(1,i) = eval(sum((y_train_result - True_output).^2)/length(x_training));
end
%figure
%plot(lamda,MSE_training)
%hold on
%plot(lamda,MSE_test)
%xlabel('Regularization')
%ylabel('MSE')
%legend('Training','Test')
%title('MSE Change with the increasing regularization')