function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%



%J = (1/2)*sum((((X*Theta') - Y).*((X*Theta') - Y))(R==1)); 
%J = (1/2)*sum((((X*Theta') - Y).^2)(R ==1)); %each element (X*Theta')-Y must be squared


J = 0.5*sum(sum(R.*(((X*Theta') - Y).^2))); %sum(sum(R.*M)) is taking sum of all elements in M which R has a 1

%Below link shows why you cant do ((X*theta)-y)'*((X*theta)-y)
%https://www.coursera.org/learn/machine-learning/discussions/all/threads/AzIrrO7wEeaV3gonaJwAFA





X_grad = ((R.*(X*Theta') - Y))*Theta;  %unreg....times by R so that only movies that have actually been rated are in included

Theta_grad = (R.*((X*Theta') - Y))'*X; %unreg....times by R so that only movies that have actually been rated are in included

Theta_reg = (lambda/2)*sum(sum(Theta.^2)); %reg term theta.....double summed like in notes
X_reg = (lambda/2)*sum(sum(X.^2)); %reg term x....double summed like in notes

%**************************************
J = J + Theta_reg + X_reg; %Cost function regulated

Xgrad_reg = lambda*X; % Xgrad reg term
Thetagrad_reg = lambda*Theta; %Thetagrad reg term

X_grad = X_grad + Xgrad_reg; %Xgrad regulated
Theta_grad = Theta_grad + Thetagrad_reg;%theta grad regulated

%..............................................









% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
