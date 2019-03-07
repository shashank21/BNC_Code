%HW2

%% Problem 1
%Create linearly separable data (arbitrarily pick 2D), with class labels of
%1 and -1 as expressed in Duda

%create a boundary line and assign associated classes to randomly generated
%vectors
w = rand(2,1);
Xmat = rand(2,100);
value1 = zeros(1,100);
for i = 1:100
    value1(1,i) = w'*Xmat(:,i);
end
threshold = mean(value1);
Xclass = zeros(1,100);
for j = 1:100
    if w'*Xmat(:,j) - threshold >= 0
        Xclass(j) = 1;
    else
        Xclass(j) = -1;
    end
end
%Save Values
save('Xmat.mat','Xmat')
save('w.mat','w')
save('Xclass.mat','Xclass')

%Create Batch Perceptron for Linear Case
load('Xmat.mat')
load('w.mat')
load('Xclass.mat')

%As described in lecture, augment X with a 1
Xmat_new1 = zeros(3,100);
Xmat_new1(1,:) = 1;
Xmat_new1(2:3,:) = Xmat;

%As described, flip all Xmat terms if Xclass is -1
Xmat_new = zeros(3,100);
for i = 1:100
    if Xclass(i) == -1
        Xmat_new(:,i) = -Xmat_new1(:,i);
    end
end

%Initialize weight vector & learning rate
iterations = 100;
w_p = zeros(3,iterations+1);
learn = 0.001;
errors = zeros(1,iterations);
%Train Batch Perceptron

for i = 1:iterations
    %Loops through Xmat examples and identifies misclassified vectors
    delta_y = 0;
    for j = 1:100
        g_x = w_p(:,i)'*Xmat_new(:,j);
        if g_x <=0 
            delta_y = delta_y + Xmat_new(:,j);
            errors(i) = errors(i) + 1;
        end
    end
w_p(:,i+1) = w_p(:,i) + learn*delta_y;

end


%Verify solution
solution = w_p(:,iterations+1);
x_example = linspace(0,1,100);
y_example = (0.0540 - 0.019.*x_example)/0.0897;
plot(x_example,y_example)
hold on

for i = 1:100
    if Xclass(i) == 1
        plot(Xmat(1,i),Xmat(2,i),'o')
        hold on
    elseif Xclass(i) == -1
        plot(Xmat(1,i),Xmat(2,i),'x')
    end
end

hold off
figure;
plot(errors)







%Create non-linearly separable dataset
Xmat_nL = rand(2,100);
w_nL = rand(2,1);
Xmat_nL = Xmat_nL.^2;

value_nL = zeros(1,100);
for i = 1:100
value_nL(1,i) = w_nL'*Xmat_nL(:,i);
end
threshold = mean(value_nL);
X_class_nL = zeros(1,100);
for j = 1:100
    if w_nL'*Xmat_nL(:,j) - threshold >= 0
        X_class_nL(j) = 1;
    else
        X_class_nL(j) = -1;
    end
end

figure;
for k = 1:100
    if X_class_nL(k) == 1
        plot(Xmat_nL(1,k),Xmat_nL(2,k),'o')
        hold on
    elseif X_class_nL(k) == -1
        plot(Xmat_nL(1,k),Xmat_nL(2,k),'x')
    end
end
hold off

%Save Values
save('Xmat_nL.mat','Xmat_nL')
save('w_nL.mat','w_nL')
save('X_class_nL.mat','X_class_nL')

%Create Batch Perceptron for Non-Linear Case

%Load Values
load('Xmat_nL.mat')
load('w_nL.mat')
load('X_class_nL.mat')

%As described in lecture, augment X with a 1
Xmat_nL_new = zeros(3,100);
Xmat_nL_new(1,:) = 1;
Xmat_nL_new(2:3,:) = Xmat_nL;

%As described in lecture, flip all Xmat terms if Xclass is -1
Xmat_nL_new1 = zeros(3,100);
for i = 1:100
    if X_class_nL(i) == -1
        Xmat_nL_new1(:,i) = -Xmat_nL_new(:,i);
    end
end

%Initialize weight vector & learning rate
iterations_nL = 100;
w_p_nL = zeros(3,iterations_nL+1);
learn_nL = 0.01;
errors_nL = zeros(1,iterations_nL);

%Train Batch Perceptron
for i = 1:iterations_nL
    %Loops through Xmat examples and identifies misclassified vectors
    delta_y_nL = 0;
    for j = 1:100
        g_x_nL = w_p_nL(:,i)'*Xmat_nL_new1(:,j);
        if g_x_nL <=0 
            delta_y_nL = delta_y_nL + Xmat_nL_new1(:,j);
            errors_nL(i) = errors_nL(i) + 1;
        end
    end
w_p_nL(:,i+1) = w_p_nL(:,i) + learn_nL*delta_y_nL;
end

%Verify solution
solution_nL = w_p_nL(:,iterations_nL+1);
x_example = linspace(0,1,100);
y_example = (0.0520 +0.1434.*x_example)/-0.073;
plot(x_example,y_example)

%% Problem 2
%Load Linear Data
load('Xmat.mat')
load('w.mat')
load('Xclass.mat')

%As described in lecture, augment X with a 1
Xmat_new = zeros(3,100);
Xmat_new(1,:) = 1;
Xmat_new(2:3,:) = Xmat;

%Find pseudoinverse as LMSE solution
Y = pinv(Xmat_new');
vec_sol = Y*Xclass';


%Verify solution-
x_example = linspace(0,1,100);
y_example = (1.8022 - 0.5683.*x_example)/2.9839;
figure;
plot(x_example,y_example)
hold on
for i = 1:100
    if Xclass(i) == 1
        plot(Xmat(1,i),Xmat(2,i),'o')
        hold on
    elseif Xclass(i) == -1
        plot(Xmat(1,i),Xmat(2,i),'x')
    end
end
hold off


%LMSE Solution for Non-Linear Data
%Load Non Linear Data
load('Xmat_nL.mat')
load('w_nL.mat')
load('X_class_nL.mat')
%As described in lecture, augment X with a 1
Xmat_nL_new = zeros(3,100);
Xmat_nL_new(1,:) = 1;
Xmat_nL_new(2:3,:) = Xmat_nL;
%Find pseudoinverse as LMSE solution
Y = pinv(Xmat_nL_new');
vec_sol_nL = Y*X_class_nL';

%Plot solution
x_example = linspace(0,1,100);
y_example = (-vec_sol_nL(1) - vec_sol_nL(2).*x_example)/vec_sol_nL(3);
figure;
plot(x_example,y_example)
hold on
for k = 1:100
    if X_class_nL(k) == 1
        plot(Xmat_nL(1,k),Xmat_nL(2,k),'o')
        hold on
    elseif X_class_nL(k) == -1
        plot(Xmat_nL(1,k),Xmat_nL(2,k),'x')
    end
end
hold off

%% Problem 3
%Initate 100 random points, uniform distribution, from 0 to 5
x_u = 10.*rand(100,1);
y_u = 10.*rand(100,1);
vec_u = zeros(2,100);
vec_u(1,:) = x_u;
vec_u(2,:) = y_u;
figure;
scatter(vec_u(1,:),vec_u(2,:))
xlabel('x1')
ylabel('x2')
title('100 Random Points with Uniform Distribution')

%Create 100 random Gaussian points
x_g = normrnd(5,5,[100,1]);
y_g = normrnd(5,5,[100,1]);
vec_g = zeros(2,100);
vec_g(1,:) = x_g;
vec_g(2,:) = y_g;
figure;
scatter(vec_g(1,:),vec_g(2,:))
xlabel('x1')
ylabel('x2')
title('100 Random Points with Gaussian Distribution')

%Create 100 random Poisson points
x_p = poissrnd(4, [100,1]);
y_p = poissrnd(4, [100,1]);
vec_p = zeros(2,100);
vec_p(1,:) = x_p;
vec_p(2,:) = y_p;
figure;
scatter(vec_p(1,:),vec_p(2,:))
xlabel('x1')
ylabel('x2')
title('100 Random Points with Poisson Distribution')

%Create weight vectors for 100 neurons
%w_n = 10.*rand(2,100);
%w_n = [zeros(1,10), ones(1,10), 2*ones(1,10), 3*ones(1,10), 4*ones(1,10), 5*ones(1,10), 6*ones(1,10),7*ones(1,10), 8*ones(1,10), 9*ones(1,10); 0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9];
%Values of w_n(i,j) don't matter because it's simply their index that
%dictates which neuron they are. only kth position values matter
w_n = rand(10,10,2);
%Do I need to define neuron position on both 3rd dimensions?
w_n_g = w_n;
w_n_p = w_n;


%Initialize SOM
%set number of iterations
iterations = 1000;
learn_u = .06;
learn_p = .1;
learn_g = .1;

%Initialize vectors necessary. Input vectors, index to randomize input
%vectors, and winner neuron vector
index1 = zeros(1,iterations);
index_p = zeros(1,iterations);
index_g = zeros(1,iterations);

input_u = zeros(2,iterations);
winner_u = zeros(2,iterations);
input_g = zeros(2,iterations);
winner_g = zeros(2,iterations);
input_p = zeros(2,iterations);
winner_p = zeros(2,iterations);

for k = 1:iterations
    
%Load random input vector
index1(1,k) = randi(100);
index_p(1,k) = randi(100);
index_g(1,k) = randi(100);
input_u(:,k) = vec_u(:,index1(1,k));
input_g(:,k) = vec_g(:,index1(1,k));
input_p(:,k) = vec_p(:,index1(1,k));

%set minimum distance high
min_dist_u = 10000;
min_dist_g = 10000;
min_dist_p = 10000;
    
%traverse each node in map, find winning node
    %Uniform Distribution 
    for i = 1:10
        for j = 1:10
        norm_node = sqrt((input_u(1,k)-i)^2 + (input_u(2,k)-j)^2); %since i,j dictate position of neuron
        if norm_node < min_dist_u
            min_dist_u = norm_node;
            winner_u(:,k) = [i;j];
        end
        end
    end
   %Gaussian Distribution 
    for i = 1:10
        for j = 1:10
        norm_node = sqrt((input_g(1,k)-i)^2 + (input_g(2,k)-j)^2); %since i,j dictate position of neuron
        if norm_node < min_dist_g
            min_dist_g = norm_node;
            winner_g(:,k) = [i;j];
        end
        end
    end
    %Poisson Distribution 
    for i = 1:10
        for j = 1:10
        norm_node = sqrt((input_p(1,k)-i)^2 + (input_p(2,k)-j)^2); %since i,j dictate position of neuron
        if norm_node < min_dist_p
            min_dist_p = norm_node;
            winner_p(:,k) = [i;j];
        end
        end
    end
    

    
    
    
%     %Gaussian Dist, old method
%     for a = 1:100
%         node_g = w_n_g(:,a);
%         norm_node_g = sqrt((input_g(1,k)-node_g(1,1))^2 + (input_g(2,k)-node_g(2,1))^2);
%         if norm_node_g < min_dist_g
%             min_dist_g = norm_node_g;
%             winner_g(:,k) = node_g;
%         end
%     end
%     %Poisson Dist, old method
%     for b = 1:100
%         node_p = w_n_p(:,b);
%         norm_node_p = sqrt((input_p(1,k)-node_p(1,1))^2 + (input_p(2,k)-node_p(2,1))^2);
%         if norm_node_p < min_dist_p
%             min_dist_p = norm_node_p;
%             winner_p(:,k) = node_p;
%         end
%     end
    
    %update weights
    %Uniform
    for a = 1:10
        for b = 1:10
        neigh_dist = sqrt((a - winner_u(1,k))^2 + (b - winner_u(2,k))^2); %needs to have a decay update
        w_n(a,b,1) = w_n(a,b,1) + (learn_u*exp(-k/iterations))*exp(-1*(neigh_dist^2)/(2*(exp(-k/iterations))^2))*(winner_u(1,k) - a); 
        w_n(a,b,2) = w_n(a,b,2) + (learn_u*exp(-k/iterations))*exp(-1*(neigh_dist^2)/(2*(exp(-k/iterations))^2))*(winner_u(2,k) - b); 
        end
    end
    %Gaussian
    for a = 1:10
        for b = 1:10
        neigh_dist = sqrt((a - winner_g(1,k))^2 + (b - winner_g(2,k))^2); %needs to have a decay update
        w_n_g(a,b,1) = w_n_g(a,b,1) + (learn_g*exp(-k/iterations))*exp(-1*(neigh_dist^2)/(2*(exp(-k/iterations))^2))*(winner_g(1,k) - a); 
        w_n_g(a,b,2) = w_n_g(a,b,2) + (learn_g*exp(-k/iterations))*exp(-1*(neigh_dist^2)/(2*(exp(-k/iterations))^2))*(winner_g(2,k) - b); 
        end
    end
    %Poisson
    for a = 1:10
        for b = 1:10
        neigh_dist = sqrt((a - winner_p(1,k))^2 + (b - winner_p(2,k))^2); %needs to have a decay update
        w_n_p(a,b,1) = w_n_p(a,b,1) + (learn_u*exp(-k/iterations))*exp(-1*(neigh_dist^2)/(2*(exp(-k/iterations))^2))*(winner_p(1,k) - a); 
        w_n_p(a,b,2) = w_n_p(a,b,2) + (learn_u*exp(-k/iterations))*exp(-1*(neigh_dist^2)/(2*(exp(-k/iterations))^2))*(winner_p(2,k) - b); 
        end
    end

 
    
%     %Gaussian, old method
%     for c = 1:100
%         neigh_dist_g = sqrt((w_n_g(1,c)-winner_g(1,k))^2 + (w_n_g(2,c)-winner_g(2,k))^2); %needs to have a decay update
%         w_n_g(:,c) = w_n_g(:,c) + (learn_g*exp(-k/iterations))*exp(-1*(neigh_dist_g^2)/(2*(exp(-k/iterations))^2))*(winner_g(:,k) - w_n_g(:,c));    
%     end
%     %Poisson, old method
%     for d = 1:100
%         neigh_dist_p = sqrt((w_n_p(1,d) - winner_p(1,k))^2 + (w_n_p(2,c) - winner_p(2,k))^2);
%         w_n_p(:,d) = w_n_p(:,d) + (learn_p*exp(-k/iterations))*exp(-1*(neigh_dist_p^2)/(2*(exp(-k/iterations))^2))*(winner_p(:,k) - w_n_p(:,d));
%     end

 end

%Plot weights
figure;
w_plot_u = reshape(w_n,[100,2]);
scatter(w_plot_u(:,1),w_plot_u(:,2));
xlabel('Weight 1')
ylabel('Weight 2')
title('Weight Distribution of SOM Neurons')
figure;
w_plot_g = reshape(w_n_g,[100,2]);
scatter(w_plot_g(:,1),w_plot_g(:,2));
xlabel('Weight 1')
ylabel('Weight 2')
title('Weight Distribution of SOM Neurons')
figure;
w_plot_p = reshape(w_n_p,[100,2]);
scatter(w_plot_p(:,1),w_plot_p(:,2));
xlabel('Weight 1')
ylabel('Weight 2')
title('Weight Distribution of SOM Neurons')

%% Problem 4

load('mnist_all.mat')

%Organize Data

%create class labels, length based on example #s from training data set

label0 = zeros(5923,10);
label0(:,1) = 1;
label1 = zeros(6742,10);
label1(:,2) = 1;
label2 = zeros(5958,10);
label2(:,3) = 1;
label3 = zeros(6131,10);
label3(:,4) = 1;
label4 = zeros(5842,10);
label4(:,5) = 1;
label5 = zeros(5421,10);
label5(:,6) = 1;
label6 = zeros(5918,10);
label6(:,7) = 1;
label7 = zeros(6265,10);
label7(:,8) = 1;
label8 = zeros(5851,10);
label8(:,9) = 1;
label9 = zeros(5949,10);
label9(:,10) = 1;
trainset = [train0;train1;train2;train3;train4;train5;train6;train7;train8;train9];
trainset(:,785) = 1;
labelset = [label0;label1;label2;label3;label4;label5;label6;label7;label8;label9];
testset = [test0;test1;test2;test3;test4;test5;test6;test7;test8;test9];
testset(:,785) = 1;

trainset = double(trainset)/255;
testset = double(testset)/255;

%Set-up MLP structure
%60000inputs + 1 bias, ~785 neurons, ~10 hidden neurons, 10 outputs
%Input vector: 60000x784 Layer1: 785x10 Hidden Layer: 10x10 Output
%Layer:10x1

lr = 0.001;
iterations = 1000;
L_one_n = 785;
H_layer_n = 10;
O_layer_n = 10;
w_L_one = 0.05 + 0.01*rand(L_one_n,H_layer_n,iterations);
w_L_h = 0.05 + 0.01*rand(H_layer_n, O_layer_n, iterations);
w_L_output = 0.05 + 0.01*rand(O_layer_n, 10, iterations);


%Define inputs and outputs of each layer
inp_L_one = trainset;
out_L_one = zeros(60000,10,iterations);
out_L_h = zeros(60000,10,iterations);
out_L_output = zeros(60000,10,iterations);
errors = zeros(60000,10,iterations);

%Run MLP on training set

delta_L_output = zeros(60000,10,iterations);
delta_L_h = zeros(60000,10,iterations);
delta_L_one = zeros(60000,10,iterations);

tic
for i = 1:iterations
    
    %Forward Pass
    out_L_one(:,:,i) = 1./(1+ exp(-(inp_L_one*w_L_one(:,:,i))));
    out_L_h(:,:,i) = 1./(1+exp(-(out_L_one(:,:,i)*w_L_h(:,:,i))));
    out_L_output(:,:,i) = 1./(1+exp(-out_L_h(:,:,i)*w_L_output(:,:,i)));
    
    %Calculate Errors
    errors(:,:,i) = labelset - out_L_output(:,:,i);
    
    %Backward Prop
    delta_L_output(:,:,i) = errors(:,:,i).*out_L_output(:,:,i).*(1-out_L_output(:,:,i));
    delta_L_h(:,:,i) = out_L_h(:,:,i).*(1-out_L_h(:,:,i)).*(delta_L_output(:,:,i)*w_L_output(:,:,i));
    delta_L_one(:,:,i) = out_L_one(:,:,i).*(1-out_L_one(:,:,i)).*(delta_L_h(:,:,i)*w_L_h(:,:,i));
    
    %Update Weights
    w_L_output(:,:,i+1) = w_L_output(:,:,i) + lr*out_L_h(:,:,i)'*delta_L_output(:,:,i);
    w_L_h(:,:,i+1) = w_L_h(:,:,i) + lr*out_L_one(:,:,i)'*delta_L_h(:,:,i);
    w_L_one(:,:,i+1) = w_L_one(:,:,i) + lr*inp_L_one'*delta_L_one(:,:,i);
     
end
toc    

err_0= zeros(iterations,1);
err_1= zeros(iterations,1);
err_2= zeros(iterations,1);
err_3= zeros(iterations,1);
err_4= zeros(iterations,1);
err_5= zeros(iterations,1);
err_6= zeros(iterations,1);
err_7= zeros(iterations,1);
err_8= zeros(iterations,1);
err_9= zeros(iterations,1);

%Get training errors
for j = 1:iterations
    err_index = errors(:,:,j);
    err_norm = vecnorm(err_index)/60000;
    err_0(j) = err_norm(1);
    err_1(j) = err_norm(2);
    err_2(j) = err_norm(3);
    err_3(j) = err_norm(4);
    err_4(j) = err_norm(5);
    err_5(j) = err_norm(6);
    err_6(j) = err_norm(7);
    err_7(j) = err_norm(8);
    err_8(j) = err_norm(9);
    err_9(j) = err_norm(10);
end

figure;
plot(1:1000, err_0,1:1000, err_1,1:1000, err_2,1:1000, err_3,1:1000, err_4,1:1000, err_5,1:1000, err_6,1:1000, err_7,1:1000, err_8,1:1000, err_9)

%Test   
T_inp_L_one = testset;
T_out_L_one = 1./(1+ exp(-(T_inp_L_one*w_L_one(:,:,iterations))));
T_out_L_h = 1./(1+exp(-(T_out_L_one*w_L_h(:,:,iterations))));
T_out_L_output = 1./(1+exp(-T_out_L_h*w_L_output(:,:,iterations))); 



