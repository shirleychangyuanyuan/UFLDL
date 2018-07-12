function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.theta 的最后一列为0
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.是个列向量，现在得把它变成矩阵
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;%theta 是n*（k-1)
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));%g是n*（k-1)，所以最后要注意选取前k-1列，去掉最后一列

%   h = theta'*X;%h(k,i)第k个theta，第i个样本  
%   a = exp(h);  
%   a = [a;ones(1,size(a,2))];%加1行  
%   p = bsxfun(@rdivide,a,sum(a));  
%   c = log2(p);  
%   i = sub2ind(size(c), y,[1:size(c,2)]);  
%   values = c(i);  
%   f = -sum(values);  
%   
%   d = full(sparse(y,1:m,1));  
%   d = d (1:num_classes-1,:); 
%   p = p(1:num_classes-1,:);;%减1行  
% g = -X * (d-p)';
%   %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
h = exp(theta' * X);
  h = [h;ones(1,m)];%加一行1是因为exp的0次为1
  p = bsxfun(@rdivide,h,sum(h));
  logp = log2(p);  
  index = sub2ind(size(p),y,1:m);%1*m，索引值
  f = -sum(logp(index));



  yk = full(sparse(y,1:m,1));
%   yk = yk(1:num_classes-1,:);
%   p = p(1:num_classes-1,:);
  g = -X * (yk-p)';
  g=g(:,1:num_classes-1);
% %   这个以验证，正确。

   
%   yCompare = full(sparse(y, 1:m, 1));  %??y == k ???  
% %yCompare = yCompare(1:num_classes-1,:); % ??y = 10???  
% M = exp(theta'*X); 
% M=[M;ones(1,m)];
% p = bsxfun(@rdivide, M, sum(M));  
% logp=log2(p);
% f = - yCompare(:)' * logp(:);  
%   
%   
% g = - X*(yCompare - p)';  
% g = g(:,1:num_classes - 1); 
%   
  
  








 g=g(:); % make gradient a vector for minFunc

