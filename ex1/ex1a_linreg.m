%
%This exercise uses a data from the UCI repository:
% Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository
% http://archive.ics.uci.edu/ml
% Irvine, CA: University of California, School of Information and Computer Science.
%
%Data created by:
% Harrison, D. and Rubinfeld, D.L.
% ''Hedonic prices and the demand for clean air''
% J. Environ. Economics & Management, vol.5, 81-102, 1978.

addpath E:\stanford_dl_ex-master\common
addpath E:\stanford_dl_ex-master\common\minFunc_2012\minFunc
addpath E:\stanford_dl_ex-master\common\minFunc_2012\minFunc\compiled

% Load housing data from file.
data = load('housing.data');%506*14的数据
data=data'; % put examples in columns 14*506 转置 每个样本是一列

% Include a row of 1s as an additional intercept feature.
data = [ ones(1,size(data,2)); data ];% 1*506的全为1的行向量，在原有的上面增加一行1

% Shuffle examples.乱序，随机的training set 和test set
data = data(:, randperm(size(data,2)));%随机打乱数据的列数，即随机取样本，行数不变

% Split into train and test sets打乱样本后，分为训练和测试集
% The last row of 'data' is the median home price.最后一行是房价
train.X = data(1:end-1,1:400);%1到14行，1到400列，即前400个样本为训练集
train.y = data(end,1:400);

test.X = data(1:end-1,401:end);%后106个样本为测试集
test.y = data(end,401:end);

m=size(train.X,2);%训练集的列数，即训练集的样本数
n=size(train.X,1);%训练集的行数，即训练集的特征（变量）数

% Initialize the coefficient vector theta to random values.
%  theta = rand(n,1);%有多少变量，产生多少theta，取0到1的随机值

% Run the minFunc optimizer with linear_regression.m as the objective.
%
% TODO:  Implement the linear regression objective and gradient computations
% in linear_regression.m 计算目标函数和梯度
%
%  tic;%开始计时
%  options = struct('MaxIter', 200);
% theta = minFunc(@linear_regression, theta, options, train.X, train.y);
% fprintf('Optimization took %f seconds.\n', toc);

% Run minFunc with linear_regression_vec.m as the objective.
%
% TODO:  Implement linear regression in linear_regression_vec.m
% using MATLAB's vectorization features to speed up your code.
% Compare the running time for your linear_regression.m and
% linear_regression_vec.m implementations.
%
% Uncomment the lines below to run your vectorized code.
%Re-initialize parameters
theta = rand(n,1);
 tic;
 theta = minFunc(@linear_regression_vec, theta, options, train.X, train.y);
 fprintf('Optimization took %f seconds.\n', toc);

% Plot predicted prices and actual prices from training set.
actual_prices = train.y;
predicted_prices = theta'*train.X;%1*m的行向量

% Print out root-mean-squared (RMS) training error.平方根误差
train_rms=sqrt(mean((predicted_prices - actual_prices).^2));
fprintf('RMS training error: %f\n', train_rms);

% Print out test RMS error
actual_prices = test.y;
predicted_prices = theta'*test.X;
test_rms=sqrt(mean((predicted_prices - actual_prices).^2));
fprintf('RMS testing error: %f\n', test_rms);

%gradient check   
average_error=grad_check(@linear_regression_vec,theta,100,train.X,train.y);  
fprintf('Average error :%g\n',average_error);  


% Plot predictions on test data.
plot_prices=true;
if (plot_prices)
  [actual_prices,I] = sort(actual_prices);
  predicted_prices=predicted_prices(I);
  plot(actual_prices, 'rx');
  hold on;
  plot(predicted_prices,'bx');
  legend('Actual Price', 'Predicted Price');
  xlabel('House #');
  ylabel('House price ($1000s)');
end