function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%PART 1

%recode y to Y

output_identity = eye(num_labels);
Y = zeros(m,num_labels);
for i = 1:m
  Y(i, :) =  output_identity(y(i),:); 
endfor



%feed forward

A1 = [ones(m, 1) X]; %add the bias(ones) to the X
Z2 = A1*Theta1';

A2 = [ones(size(Z2), 1) sigmoid(Z2)];

Z3 = A2*Theta2';

A3 = sigmoid(Z3);

h = A3; %A3 = hypothesis

%J= (1/m)*trace((-Y'*log(h) - (1-Y)'*log(1-h)).*eye(num_labels));%lmatrix version of cost function. Need to make Y*h into a KxK matrix. Then write the cost function as normal. ELEMENT wise multiply by K-sized identity matrix. take the trace. Multiply by 1/m

%Add regularisation

theta1_temp = Theta1;
theta1_temp(:,1) = 0; %make first column of theta1 zeroes so as not to affect bias

theta2_temp = Theta2;
theta2_temp(:,1) = 0; %make first column of theta2 zeroes so as not to affect bias


theta1sq = (theta1_temp'*theta1_temp);
theta1reg = trace(theta1sq.*eye(size(theta1sq)));  %getting theta1 regularisation term

theta2sq = (theta2_temp'*theta2_temp);
theta2reg = trace(theta2sq.*eye(size(theta2sq))); %getting theta2 regularisation term

%Put J + regularisation

%J = (1/m)*sum(sum((-Y'*log(h) - (1-Y)'*log(1-h)).*eye(num_labels))) + (lambda/(2*m))*(theta1reg+theta2reg);

J= (1/m)*trace((-Y'*log(h) - (1-Y)'*log(1-h))) + (lambda/(2*m))*(theta1reg+theta2reg); %Above and below are two different zays of the same code

%Link for trace and sum(sum()
%https://www.coursera.org/learn/machine-learning/discussions/all/threads/AzIrrO7wEeaV3gonaJwAFA

%****************************
%****************************

%PART 2

%         Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

%Backpropagation

Delta1 = zeros(size(Theta1)); %make Delta1 size of Theta1 for later
Delta2 = zeros(size(Theta2)); %male Delta2 size of Theta2 for later

Xones = [ones(m, 1) X];%add the bias(ones) to the X
for t= 1:m
 %forward prop 
a1 = Xones(t,:); 
#size(a1)
#size(Theta1)
z2 = Theta1*a1'; %needs to be a vector not a row so instead pf a1*Theta1' it is Theta1*a1'
#size(Z2)

a2 = [1 ; sigmoid(z2)];

z3 = Theta2*a2; %a2 is already a column vector so does not need to be transposed


a3 = sigmoid(z3); 

%getting D3 and D2
 
d3 = a3 - (Y(t, :))'; 

z2 = [1;z2]; %have to put bias back so that element wise multiplication with sigmoidGradient(z2) is dimensionally correct

d2 = Theta2'*d3.*sigmoidGradient(z2);

d2 = d2(2:end); %remove bias node
 
%Update gradients
 
Delta1 = Delta1 + d2*a1;
Delta2 = Delta2 + d3*a2'; %transpose for dimensionally correct multiplication
endfor



%Add regularisation and create gradient


%Theta1_grad = Delta1/m; %unregularized gradient
%Theta2_grad = Delta2/m;%unregularized gradient

theta1_temp = Theta1;
theta1_temp(:,1) = 0; %makes zeroes in first column so as to not to affect bias unit

theta2_temp = Theta2;
theta2_temp(:,1) = 0; %makes zeroes in first column so as to not to affect bias unit


Theta1_grad = Delta1/m + (lambda/m)*theta1_temp; %regularized gradient for theta 1 layer
Theta2_grad = Delta2/m + (lambda/m)*theta2_temp;%regularized gradient for theta 2 layer

%*****************
%*****************















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
