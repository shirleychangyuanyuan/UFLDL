function [f,g] = logistic_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %

  m=size(X,2);
  n=size(X,1);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));


  %
  % TODO:  Compute the objective function by looping over the dataset and summing
  %        up the objective values for each example.  Store the result in 'f'.
  %
  % TODO:  Compute the gradient of the objective by looping over the dataset and summing
  %        up the gradients (df/dtheta) for each example. Store the result in 'g'.
  %
%%% YOUR CODE HERE %%%
 h=sigmoid(theta'*X);
 f=-y*log(h')+(y-1)*log(1-h');
 g=X*(h-y)';
% for i=1:m
%     f=f-(y(i)*log(sigmoid(theta'*X(:,i)))+(1-y(i))*log(1-sigmoid(theta'*X(:,i))));
% end
% 
% 
% for j=1:n
% for i=1:m
%     g(j)=g(j)+X(j,i)*(sigmoid(theta'*X(:,i))-y(i));
% end
% end


