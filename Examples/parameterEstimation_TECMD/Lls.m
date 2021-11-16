function [theta,results] = Lls(K,Z,varargin)
%
% Computes a numerically stable (weighted) linear leaast squares estimate
% and returns the optimal estimate, its variance and an estimate of the
% noise variance. Parameters will be non-unique if K is rank deficient.
%
% Inputs (mandatory):
%   K: Regresor matrix, size n x m
%   Z: Output vector, size n x 1
%
% Inputs (optional):
%   ny: Output noise vector, std of output noise for weighted least
%   squares, size n x 1. default ny = ones(n,1);
%   plotFit: Set plot as 1 or zero to look at the fitted vs data
%
% Outputs:
%   theta: Optimum parameter estimate, size m x 1
%   results: Structure variable with following fields
%            - paraVar: Variance estimate of theta, size m x 1
%            - paraCov: Parameter covariance matrix size m x m
%            - noiseVar: Estimate of noise variance, size 1 x 1
%            - resVec: Residual vector, size n x 1
%            - resNorm: 2 norm of residual vector, size 1 x 1
%            - regMsg: Message stating rank of regressor
%            - regRankFull: A flag with value 0 or 1 if regressor is rank deficient or not.
%
% Copyright (C) W. D. Widanage -  WMG, University of Warwick, U.K. 11-02-2012-12/01/2016 (Through the never)
% All Rights Reserved
% Software may be used freely for non-comercial purposes only

Z = Z(:); % Vectorise
parObj = inputParser; % Create an input parse object to handle positional and property-value arguments

% Create variable names and assign default values after checking the value
addRequired(parObj,'K', @isnumeric);
addRequired(parObj,'Z', @isnumeric);

% Optional parameters
addOptional(parObj,'ny',ones(size(Z)),@isnumeric);
addOptional(parObj,'plotFit',0,@isnumeric);



% Re-parse parObj
parse(parObj,K,Z,varargin{:})


K = parObj.Results.K;
Z = parObj.Results.Z;
ny = parObj.Results.ny;
plotFit = parObj.Results.plotFit;

% Weight matrix
W = spdiags(1./ny,0,length(ny),length(ny));

%Multiply by weights
Z = W*Z;
K = W*K;

%Normalise K with the 1/norm of each column
Knorm = sqrt(sum(abs(K).^2));
idxZeros = Knorm<1E-14;
Knorm(idxZeros) = 1;
N = diag(1./Knorm);
Kn = K*N;

%Compute Lls via economic SVD decompostion
[U, S, V] = svd(Kn,0);    % Perform SVD
ss = diag(S);             % Singular values
idxZeros = ss < 1E-14;    % Replace any small singular value with inf
nCol = size(Kn,2);
if sum(idxZeros)>0        % If there are zero singular values sum(idxZeros) > 0 and regressor is rank deficient
    results.regMsg = sprintf('Estimated parameters are NON-UNIQUE.\n Lls regressor is rank defficient. Rank = %d instead of %d. \nParameters estimated from a subspace.',nCol-sum(idxZeros),nCol);
    fprintf('\nEstimated parameters are NON-UNIQUE.\nRegressor rank defficient. Rank = %d instead of %d. \nParameters estimated from a subspace.\n\n',nCol-sum(idxZeros),nCol)
    results.regRankFull = 0;
else
    results.regMsg = sprintf('Estimated parameters are unique.\nRegressor preserves full rank. Rank =  %d.',nCol);
    results.regRankFull = 1;
end
ss(idxZeros) = inf;
Sv = diag(1./ss);               %Inverse singular value matrix

% Least squares solution
theta = N*V*Sv*U'*Z;
% cond(N*V*Sv*U')

%Projection matrix and residuals
P = (eye(length(Z))-U*(U')); % Projection matrix
R = P*Z;                     % Residuals

%Estimate of noise and parameter variance
if ismember('ny',parObj.UsingDefaults)                      % If measurement variance is not provided estiamte via residuals
    results.noiseVar = ((R')*R)/real(trace(P));             % Noise variance
    results.paraCov = (N')*V*Sv*Sv*(V')*N*results.noiseVar; % Parameter covariance
else                                                        % Else no need to estimate noise variance
    results.noiseVar = [];                                  % Noise variance
    results.paraCov = (N')*V*Sv*Sv*(V')*N;                  % Parameter covariance    
end

if plotFit
    figure
    Zm = K*theta;
    plot([1:length(Z)]',Zm,'. -',[1:length(Z)]',Z,'o-');
    xlabel('Index (-)'); ylabel('Model and Measured')
    legend('Model','Data')
end
results.paraVar = diag(results.paraCov);                % Parameter variance
results.resVec = R;                                     % Residual vector
results.resNorm = norm(R);                              % Residual norm

