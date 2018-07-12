function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.theta �����һ��Ϊ0
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.�Ǹ������������ڵð�����ɾ���
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;%theta ��n*��k-1)
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));%g��n*��k-1)���������Ҫע��ѡȡǰk-1�У�ȥ�����һ��

%   h = theta'*X;%h(k,i)��k��theta����i������  
%   a = exp(h);  
%   a = [a;ones(1,size(a,2))];%��1��  
%   p = bsxfun(@rdivide,a,sum(a));  
%   c = log2(p);  
%   i = sub2ind(size(c), y,[1:size(c,2)]);  
%   values = c(i);  
%   f = -sum(values);  
%   
%   d = full(sparse(y,1:m,1));  
%   d = d (1:num_classes-1,:); 
%   p = p(1:num_classes-1,:);;%��1��  
% g = -X * (d-p)';
%   %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
h = exp(theta' * X);
  h = [h;ones(1,m)];%��һ��1����Ϊexp��0��Ϊ1
  p = bsxfun(@rdivide,h,sum(h));
  logp = log2(p);  
  index = sub2ind(size(p),y,1:m);%1*m������ֵ
  f = -sum(logp(index));



  yk = full(sparse(y,1:m,1));
%   yk = yk(1:num_classes-1,:);
%   p = p(1:num_classes-1,:);
  g = -X * (yk-p)';
  g=g(:,1:num_classes-1);
% %   �������֤����ȷ��

   
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

