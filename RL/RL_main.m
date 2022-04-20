clc; clear; close all; load("qeval.mat")
%% Obtain a new reward function
% Obtained a new reward function by shuffling the original reward function
% row-wise. Do not run this section as it would change the results written
% in the report
%load("task1.mat")
%reward_sub1 = reward(1:89,:);
%reward_sub2 = reward(90:100,:);
%qevalreward = reward_sub1(randperm(size(reward_sub1, 1)), :);
%qevalreward = vertcat(qevalreward,reward_sub2);
%save("qeval.mat","qevalreward")
%% RL Variables
% Grid map & starting/goal states,indexes
grid_map = reshape(1:1:100,[10 10]);
initial_s = 1; initial_m=1; initial_n=1; 
final_s = 100; final_m=10; final_n=10;

threshold = 5E-3;       
reward_info = qevalreward;
t = 0; % Time
%% RL Algorithm
reached_goal = 0; avg_t = 0; max_reward = 0;
q_grid = cell(10,10); negative_r = -inf;
param=input('Enter a number (1-3) to choose one among the parameter values shown below: ');
dr=0.9; % Discount factor
% 1) param = exp(-0.001*k);
% 2) param = exp(-0.0015*k);
% 3) param = exp(-0.002*k);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for run = 1:10
    tic
    disp(['Attempted Run: ',num2str(run)]);
    %%%%%%% 1.Initial Reward_Grid Map and Initial Q-function Values%%%%%%%%
    state = 1;
    q_val = zeros(100,4);
    for i = 1:size(reward_info,1)
        for j = 1:size(reward_info,2)
            if reward_info(i,j) == -1
                q_val(i,j) = negative_r;
            else
                q_val(i,j) = 0;
            end
        end
    end
    row = 1;
    for i = 1:10
        for j = 1:10
            q_grid{j,i} = q_val(row,:);
            row = row + 1;
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%% 2. Q-Learning Algorithm %%%%%%%%%%%%%%%%%%%%%%%%%
    trial = 1;
    convergence = false;
    while trial <= 3000 && ~convergence
        new_q_grid = q_grid;
        k = 1;
        state = initial_s; alpha = threshold + 1;
        
        while state~=final_s && alpha > threshold
            %%%%%%%%%%%%%%%%%%% Parameter Selection %%%%%%%%%%%%%%%%%%%%%%%
            if param == 1
                alpha = exp(-0.001*k);
            elseif param == 2
                alpha = exp(-0.0015*k);
            elseif param == 3
                alpha = exp(-0.002*k);
            end
           
            if alpha > 1 || alpha == inf || alpha == -inf
                alpha = 1;
            end         
            explore = alpha; % "alpha == explore" in this project
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [m, n] = find(grid_map == state); % Current state position
            q_values = q_grid{m,n}; % Q-values of the actions in the state
            
            %%%%%%%%%%%%%%%%%%% Probability Calculation %%%%%%%%%%%%%%%%%%%
            probability = [inf inf inf inf];

            max_value = max(q_values);
            max_value_idx = find(max_value == q_values);
            max_q_idx = [];

            if length(max_value_idx) ~= 1
                max_q_idx = max_value_idx(randi(length(max_value_idx)));
            end

            invalid_idx = find(q_values == negative_r);
            probability(1, invalid_idx) = 0; 

            probability(1,max_q_idx) = 1 - explore;

            etc_idx = find(inf == probability);
            num_etc = length(etc_idx);
            probability(1,etc_idx) = explore / num_etc;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
            %%%%%%%%%%%%%%%%%%%%%%%%% Next state %%%%%%%%%%%%%%%%%%%%%%%%%%
            action = randsample(4,1,true,probability); % Action to move to the next state
            switch action
                case 1
                    if state==1
                        action = randsample([2 3],1,true,probability(1,2:3));
                        if action == 2
                            next_state = state + 10;
                        elseif action == 3
                            next_state = state + 1;
                        end
                    elseif state==11||state==21||state==31||state==41||...
                            state==51||state==61||state==71||state==81
                        action = randsample([2 3 4],1,true,probability(1,2:4));
                        if action == 2
                            next_state = state + 10;
                        elseif action == 3
                            next_state = state + 1;
                        elseif action == 4
                            next_staet = state - 10;
                        end
                    elseif state==91
                        action = randsample([3 4],1,true,probability(1,3:4));
                        if action == 3
                            next_state = state + 1;
                        elseif action == 4
                            next_state = state - 10;
                        end
                    else
                        next_state = state - 1;
                    end
                case 2
                    if state > 90
                        action = randsample([1 3 4],1,true,[probability(1,1) probability(1,3:4)]);
                        if action == 1
                            next_state = state - 1;
                        elseif action == 3
                            next_state = state + 1;
                        elseif action == 4
                            next_state = state - 10;
                        end
                    else
                        next_state = state + 10;
                    end
                case 3
                    if state==10||state==20||state==30||state==40||state==50||...
                            state==60||state==70||state==80||state==90
                        action = randsample([1 2 4],1,true,[probability(1,1:2) probability(1,4)]);
                        if action == 1
                            next_state = state - 1;
                        elseif action == 2
                            next_state = state + 10;
                        elseif action == 4 && state ~= 10
                            next_state = state - 10;
                        end
                    else
                        next_state = state + 1;
                    end
                case 4
                    if state < 11
                        action = randsample(3,1,true,probability(1,1:3));
                        if action == 1 && state ~= 1
                            next_state = state - 1;
                        elseif action == 2
                            next_state = state + 10;
                        elseif action == 3
                            next_state = state +1;
                        end
                    else
                        next_state = state - 10;
                    end
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%%%%%%%%%%%%%%%%%%% Q-function Iteration %%%%%%%%%%%%%%%%%%%%%
            r = reward_info(state,action);
            [a, b] = find(next_state == grid_map);
            future_q = q_grid{a,b};
            q_values(1,action) = q_values(1,action) + alpha * (r + dr * max(future_q) - q_values(1,action) );
            q_grid{m,n} = q_values;
            state = next_state; k=k+1;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%% 3.Convergence Check %%%%%%%%%%%%%%%%%%%%%%%%%%
        check = false;
        diff = cellfun(@minus,new_q_grid,q_grid,'UniformOutput',false);
        for i = size(diff,1)
            for j = size(diff,2)
                temp(i,j) = max(abs(diff{i,j}));
            end
        end
    if mod(trial,3000) == 0
        if max(max(temp)) < threshold
            disp('Convergence Completed')
            check = true;
        end
    end
        trial = trial + 1;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%% 4. Tracking the vectors (Reward&Optimal Path)%%%%%%%%%%%%%
    idx1 = initial_m; idx2 = initial_n;
    action_vec = {}; qevalstates = []; move_vec = []; action_numeric = [];
    reward_grid = zeros(10,10); 
    while grid_map(idx1,idx2)~=final_s && reward_grid(idx1,idx2)==0
        [~, action_val] = max(q_grid{idx1,idx2});
        if action_val == 1
            action = '^';
            action2 = 1;
            new_idx1 = idx1 - 1;
            new_idx2 = idx2;
        elseif action_val == 2
            action = '>';
            action2 = 2;
            new_idx1 = idx1;
            new_idx2 = idx2 + 1;
        elseif action_val == 3
            action = 'v';
            action2 = 3;
            new_idx1 = idx1 + 1;
            new_idx2 = idx2;
        elseif action_val == 4
            action = '<';
            action2 = 4;
            new_idx1 = idx1;
            new_idx2 = idx2 - 1;
        end
        
        reward_grid(idx1,idx2) = reward_info(grid_map(idx1,idx2),action_val);
        action_numeric = vertcat(action_numeric,action2);
        action_vec = vertcat(action_vec,strcat(action,'b'));
        qevalstates = vertcat(qevalstates,grid_map(idx1,idx2));
        move_vec = horzcat(move_vec,[idx1-0.5;idx2-0.5]); % "-0.5" for plotting properly

        if new_idx1==final_m && new_idx2==final_n
            action_vec = vertcat(action_vec,'rp');
            qevalstates = vertcat(qevalstates,100);
            move_vec = horzcat(move_vec,[9.5;9.5]);
        end
        idx1 = new_idx1;
        idx2 = new_idx2;
    end
    r = sum(reward_grid,'all');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%% 5. Time calculation %%%%%%%%%%%%%%%%%%%%%%%%%%%
    if qevalstates(end,1) == final_s
        t = t + toc;
        reached_goal = reached_goal + 1;
        avg_t = t/reached_goal;
        tot_avg_t = t/run;
        disp(['Successful arrival at the goal state',newline])
    else
        t = t + toc;
        tot_avg_t = t/run;
        disp(['Failed to reach the goal state',newline])
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
end
%%%%%%%%%%%%%%%%%%%%%%%%%% 5. Optimal Policy %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Optimal_map = cell(10,10);
for i = 1:10
    for j = 1:10
        if i == 10 && j == 10
            Optimal_map{10,10} = 0;
        else      
            Optimal_map{i,j} = find(q_grid{i,j}==max(q_grid{i,j}));
        end
    end
end
opt_p = cell(10,10);
for i = 1:10
    for j = 1:10
        if Optimal_map{i,j} == 1
            opt_p{i,j} = '^';
        elseif Optimal_map{i,j} == 2
            opt_p{i,j} = '>';
        elseif Optimal_map{i,j} == 3
            opt_p{i,j} = 'v';
        elseif Optimal_map{i,j} == 4
            opt_p{i,j} = '<';
        elseif Optimal_map{i,j} == 0
            opt_p{i,j} = 'p';
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%% 6.Total Reward %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1 : length(qevalstates)-1
    max_reward = max_reward + dr^k * reward_info(qevalstates(k,1),action_numeric(k)); % Gamma = 0.9
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Results
disp(['Reached the goal state ', num2str(reached_goal),' times of 10 run'])
disp(['Total convergence time: ' num2str(tot_avg_t)]);
disp(['Maximum reward: ' num2str(max_reward)]);
disp(['Average time: ', num2str(avg_t)])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot (Optimal Policy)
figure(1)
xlim([0,10]);ylim([0,10]);axis ij; title("Optimal Policy Map")
grid on; hold on;
opt_p2 = strcat(opt_p,'b');
grid_row = [0.5:1:9.5];
grid_col = [0.5:1:9.5];
for i = 1:10
    for j = 1:10
        plot(grid_col(j),grid_row(i),opt_p2{i,j},'MarkerSize',12);
    end
end

%% Plot (Optimal Path)
figure(2)
xlim([0,10]);ylim([0,10]);axis ij; title("Optimal Path Map")
grid on; hold on;
for i = 1:size(move_vec,2)
    plot(move_vec(2,i),move_vec(1,i),action_vec{i,1},'MarkerSize',12);
end
xlabel(['Total reward = ', num2str(max_reward)])