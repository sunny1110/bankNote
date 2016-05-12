function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)






Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% Calculating h, by considering the network layers

a1 = [ones(size(X), 1), X]; %Adding ones to X
z2 = a1 * transpose(Theta1);
a2 = sigmoid(z2);
a2= [ones(size(a2),1), a2]; %Adding ones to a2
z3 = a2 * transpose(Theta2);
a3 = sigmoid(z3);
h = a3;

% The resulting value of h will be of the following dimensions:
% 5000, 10
% The above me seem like an extremely complicated dimension set for a hypothesis but it is a classification problem.
% Dimensions of X (before bias) : m X 400.
% Dimensions of X (after bias) : m X 401.
% Dimensions of Theta1: 401 X 25, where 401 is input units size (bias included), and 25 is hidden units size.
% Dimensions of a2 (before bias) : m X 25.
% Dimensions of a2 (after bias) : m X 26.
% Dimensions of Theta2: 26 X 10, where 26 is the hidden units size (bias included), and 10 is the Hypothesis units size

% Calulations of Costs will follow:
y_matrix = eye(num_labels)(y,:);



pdiff = log(h) .* y_matrix;
ndiff = log(1-h) .* (1-y_matrix);
diff = pdiff + ndiff;
rowsum = sum(diff, 2); % Size: 5000 X 1
colsum = sum(rowsum); % Size: 1 X 1
J = -colsum /m;
NT1 =Theta1(1:size(Theta1, 1), 2:size(Theta1, 2));
NT2 =Theta2(1:size(Theta2, 1), 2:size(Theta2, 2));
R = (sum(NT1(:).^2) + sum(NT2(:).^2))/(2*m) * lambda;
J = J + R;


% The implementation for backpropagation will now begin.
% This is a higly vectorised solution, hence the solutions might appear complicated.


% This is the difference matrix, from which we obtain delta3 from.
d3 = a3 - y_matrix;

% This gives us delta2, from which we obtain delta2 from.
% TEMPORARILY HIDING THIS CODE d2 =(d3 * Theta2(:, 2:end)) .* sigmoidGradient(z2);
% a1(:, 2:end), this is a1 without the Bias.
d2 =(d3 * Theta2(:, 2:end)) .* sigmoidGradient(z2);
Delta1 = transpose(d2) * a1;
Delta2 = transpose(d3) * a2;

% ([zeros(size(Theta1,2),1)'; Theta1])
% The above statement appends a row of zeros to make regularization easier.


Theta1_grad = Delta1 ./m .+ ([zeros(size(Theta1(:, 2:end),1),1), Theta1(:, 2:end)] .* (lambda ./m));

Theta2_grad = Delta2 ./m .+ ([zeros(size(Theta2(:, 2:end),1),1), Theta2(:, 2:end)] .* (lambda ./m));







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
