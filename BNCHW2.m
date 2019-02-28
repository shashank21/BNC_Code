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

%Create 100 random Gaussian points
x_g = normrnd(2.5,1,[100,1]);
y_g = normrnd(2.5,1,[100,1]);
vec_g = zeros(2,100);
vec_g(1,:) = x_g;
vec_g(2,:) = y_g;
figure;
scatter(vec_g(1,:),vec_g(2,:))

%Create 100 random Poisson points
x_p = poissrnd(4, [100,1]);
y_p = poissrnd(4, [100,1]);
vec_p = zeros(2,100);
vec_p(1,:) = x_p;
vec_p(2,:) = y_p;
figure;
scatter(vec_p(1,:),vec_p(2,:))

%Create weight vectors for 100 neurons
%w_n = 10.*rand(2,100);
w_n = [zeros(1,10), ones(1,10), 2*ones(1,10), 3*ones(1,10), 4*ones(1,10), 5*ones(1,10), 6*ones(1,10),7*ones(1,10), 8*ones(1,10), 9*ones(1,10); 0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9];
w_n_g = w_n;
w_n_p = w_n;
figure;
scatter(w_n(1,:),w_n(2,:))

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
    for i = 1:100
        node = w_n(:,i);
        norm_node = sqrt((input_u(1,k)-node(1,1))^2 + (input_u(2,k)-node(2,1))^2);
        if norm_node < min_dist_u
            min_dist_u = norm_node;
            winner_u(:,k) = node;
        end
    end
    %Gaussian Dist
    for a = 1:100
        node_g = w_n_g(:,a);
        norm_node_g = sqrt((input_g(1,k)-node_g(1,1))^2 + (input_g(2,k)-node_g(2,1))^2);
        if norm_node_g < min_dist_g
            min_dist_g = norm_node_g;
            winner_g(:,k) = node_g;
        end
    end
    %Poisson Dist    
    for b = 1:100
        node_p = w_n_p(:,b);
        norm_node_p = sqrt((input_p(1,k)-node_p(1,1))^2 + (input_p(2,k)-node_p(2,1))^2);
        if norm_node_p < min_dist_p
            min_dist_p = norm_node_p;
            winner_p(:,k) = node_p;
        end
    end
    
    %update weights
    %Uniform
    for j = 1:100
        neigh_dist = sqrt((w_n(1,j)-winner_u(1,k))^2 + (w_n(2,j)-winner_u(2,k))^2); %needs to have a decay update
        w_n(:,j) = w_n(:,j) + (learn_u*exp(-k/iterations))*exp(-(neigh_dist^2)/(2*(exp(-k/iterations))^2))*(winner_u(:,k) - w_n(:,j)); %change learn_u to learn_u(k)
    end
    %Gaussian
    for c = 1:100
        neigh_dist_g = sqrt((w_n_g(1,c)-winner_g(1,k))^2 + (w_n_g(2,c)-winner_g(2,k))^2); %needs to have a decay update
        w_n_g(:,c) = w_n_g(:,c) + (learn_g*exp(-k/iterations))*exp(-(neigh_dist_g^2)/(2*(exp(-k/iterations))^2))*(winner_g(:,k) - w_n_g(:,c));    
    end
    %Poisson
    for d = 1:100
        neigh_dist_p = sqrt((w_n_p(1,d) - winner_p(1,k))^2 + (w_n_p(2,c) - winner_p(2,k))^2);
        w_n_p(:,d) = w_n_p(:,d) + (learn_p*exp(-k/iterations))*exp(-(neigh_dist_p^2)/(2*(exp(-k/iterations))^2))*(winner_p(:,k) - w_n_p(:,d));
    end
end

%Plot weights
figure;
scatter(w_n(1,:),w_n(2,:));
figure;
scatter(w_n_g(1,:),w_n_g(2,:));
figure;
scatter(w_n_p(1,:),w_n_p(2,:));

%% Problem 4


