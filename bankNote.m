%% Initialization
clear ; close all; clc

fprintf('\nStart time: ');
disp(ctime(time()));
fprintf('\n');
start_time = clock();
fprintf('\nThe Bank Note detection Algorithm\n');


%% Setting up the necessary parameters

hidden_layer_size = 3; % 21 hidden units
num_labels = 2; % 10 labels for each digit 0-9
lambdaSet = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 9, 27, 81, 243]; % Regularization parameters
cluster = zeros(1, 4);


%% Loading up of Data

fprintf("\nLoading data...\n");

load('bankNoteData.csv'); %loading of the training data

fprintf("\nDone!\n");

% Duplicating X

dupX = X;

% Scaling features

fprintf('\nScaling Features...\n');

for i = 1:size(dupX, 2)
  X(:, i) = (dupX(:, i) .- min(dupX(:, i))) ./ (max(dupX(:, i)) .- min(dupX(:, i)));
end

fprintf('\nScaled!\n');






%% Dividing the data.

fprintf('\nDividing the data set....\n');
mat = randperm(size(X, 1));

X_train = X(mat(1:878), :);
y_train = y(mat(1:878), :);
X_cv = X(mat(879:1098), :);
y_cv = y(mat(879:1098), :);
X_test = X(mat(1099:end), :);
y_test = y(mat(1099:end), :);

fprintf('\nDone!\n')

input_layer_size = size(X_train, 2);
m = size(X_train, 1);






% Initialise weights to valid range

fprintf('\nInitialising Neural Network Weights...\n');
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


% Training the Neural Network now Begins.

fprintf('\nThe training of the Neural Network shall now begin...\n')

options = optimset('MaxIter', 150);
for i = 1:size(lambdaSet, 2)

  lambda = lambdaSet(i);

  printf('\nTraining with Lambda = %f\n', lambda);
  costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda);
  [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size +1)), hidden_layer_size, (input_layer_size + 1));
  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size +1))):end), num_labels, (hidden_layer_size + 1));
  pred_train = predict(Theta1, Theta2, X_train);
  accuracy_train = (mean(double(pred_train == y_train)) * 100);
  pred_cv = predict(Theta1, Theta2, X_cv);
  accuracy_cv = (mean(double(pred_cv == y_cv)) *100);
  pred_test = predict(Theta1, Theta2, X_test);
  accuracy_test = (mean(double(pred_test == y_test)) *100);
  cluster = [cluster; lambda, accuracy_train, accuracy_cv, accuracy_test];
  printf('\nAccuracy results are as follows:\nTraining: %f\nCrossValidation: %f\nTest: %f\n\n',accuracy_train, accuracy_cv, accuracy_test);
 end


disp(cluster);

save holyGrail.csv cluster;
fprintf('\nEverything Done!\n')
disp(ctime(time()));
end_time = clock();
elapsed_time = etime(end_time, start_time);
printf('\nTime taken for completion is : %f\n',elapsed_time);
