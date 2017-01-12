function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
% %%% YOUR CODE HERE %%%

% This is the loop version
%   for i=1:m
%      denominator = sum(exp(theta'*X(:,i)))+1;
%      if y(i)==10
%         f = f - log(1/denominator);
%      else
%         f = f - log(exp(theta(:,y(i))'*X(:,i))/denominator);
%      end
% 
%      for k=1:num_classes-1
%         if y(i)==k
%            g(:,k) = g(:,k) - X(:,i)*(1 - exp(theta(:,k)'*X(:,i))/denominator);
%         else
%            g(:,k) = g(:,k) - X(:,i)*(0 - exp(theta(:,k)'*X(:,i))/denominator);
%         end
%      end
%   end
  
% This is the vector version
  M = exp(theta'*X);
  % denominator'size is 1*60000
  denominator = sum(M,1);
  % numerator's size is 10*60000
  numerator = [M;ones(1,size(denominator,2))];
  % fraction's size is 10*60000
  fraction = bsxfun(@rdivide, numerator, denominator);
  % afterlog's size is 60000*10
  afterlog = log(fraction)';
  
  I=sub2ind(size(afterlog), 1:size(afterlog,1), y);
  values = afterlog(I);
  f = -sum(values);
  
  y_full = full(sparse(y,1:m,1));
  g = -X*(y_full-fraction)';
  g = g(:,1:size(g,2)-1);
  g=g(:); % make gradient a vector for minFunc
  
  

