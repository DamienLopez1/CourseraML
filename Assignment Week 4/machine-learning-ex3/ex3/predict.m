function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

A1 = [ones(m, 1) X]; %add the bias(ones) to the X
Z2 = A1*Theta1'; % to get a 5000 x 25 matrix mapping inputs to hidden layer

A2 = sigmoid(Z2); %perfroms squishing function

A2ones = [ones(size(A2), 1) A2]; %ads bias onto hidden layer

Z3 = A2ones*Theta2'; %maps hidden layer to output layer

A3 = sigmoid(Z3); %appliess squishing function

[predict_max, index_max] = max(A3, [], 2); %gets index of which label has highest value

p = index_max; %index is directly related to which number the character is




% =========================================================================


end
