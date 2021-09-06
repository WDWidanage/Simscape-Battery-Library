function [theta,results] = LMAlgorithm(fh, theta0, u, d, varargin)
%
% Perform nonlinear least squares optimisation using the
% Levenberg-Marquardt method
%
% Minimised cost-function f(theta)
%   f(theta) = sum ((G(theta)_i-d_i)/s_i)^2  i = 1...M
%
% Mandotory input argumetns
%   fh: function handle of nonlinear model. fh = @(theta, u)fcn(theta,u,optArg1,...,optArgN).
%       Nonlinear model function should return two outputs, the model output and Jacobian as second
%   theta0: Initial starting point of model parameters, size N x 1
%   u: Input signal to simulate nonlinear model, size Mu x 1
%   d: Measured output data, size M x 1
%
% Optional arguments. Create a structure variable with the following fields:
%   Jacobian: Specify as 'on' if model function returns the Jacobian or to 'off' to approximate by finte forward difference, default Jacobian ='off'.
%   s: Residual weights, normally std of noise, default s = 1, size M x 1
%   iterMax: Maximum number of iterations, default iterMax = 1000, double size 1 x 1
%   TolDT:  Termination tolerance of parameter update, default TolDT = 1E-6, double size 1 x 1
%   diagnosis: Set daignosis to 'on' to plot cost-function and lambda vs iterations, default diagnosis = 'off'
%   epsilon: if Jacobian is 'off', use epsilon for parameter incerement to calculate approximate Jacobian, default epsilon = 1E-6, double size 1 x 1
%   dispMsg: Specifiy as 'on' or 'off'. If 'on' messages will be printed 
%            else no messages are printed. Default 'on'.
%
% Output arguments:
%   theta: Optimised parameter vector, size N x 1
%   results: Sturcture variable with fields
%            - covTheta: Covariance matrix of optimum parameters, size N x N
%            - stdTheta: Standard deviation of optimum parameters, size N x 1
%            - fracEr: Fractional error of optimum parameters, size N x 1
%            - cF_iter: cost-function value at each iteration 
%            - L_iter: Lambda weight at each iteration 
%            - termMsg: Reason for Levenberg-Marquardt iteration termination
%            - rankMsg: Message stating rank of Levenberg-Marquardt regressor for each iteration, size ~ x 1
%            - LMRankFull: A flag at each iteration with value 0 or 1 if regressor is rank deficient or not, size ~ x 1
%
% Copyright (C) W. D. Widanage -  WMG, University of Warwick, U.K. 14/10/2015 (Highway to hell!!)
% All Rights Reserved
% Software may be used freely for non-comercial purposes only


p = inputParser; % Create an input parse object to handle positional and property-value arguments
theta0 = theta0(:);
nData = length(d);
nPara = length(theta0);

% Create variable names and assign default values after checking the value
addRequired(p,'fh', @checkFunctionHandle);
addRequired(p,'theta0', @isnumeric);
addRequired(p,'u', @isnumeric);
addRequired(p,'d', @isnumeric);

% Optional parameters
addParameter(p,'Jacobian','off')
addParameter(p,'s',ones(size(d)),@checkS)
addParameter(p,'iterMax',1000,@isnumeric)
addParameter(p,'TolDT',1E-6,@isnumeric)
addParameter(p,'diagnosis','off')
addParameter(p,'epsilon',1E-6,@isnumeric)
addParameter(p,'dispMsg','on')

% Re-parse parObj
parse(p,fh, theta0, u, d, varargin{:})

% If s is passed as a null vector revert to default
if isempty(p.Results.s)
    varargin{1,1} = rmfield(varargin{1,1},'s');
    parse(p,fh, theta0, u, d, varargin{:})
end

% Initialise
theta_prev = p.Results.theta0;
JacStatus = lower(p.Results.Jacobian);
Jw = spdiags(1./p.Results.s,0,nData,nData); % Residual weights (noise std) of cost-function to scale each row of the Jacobain 


% Evalaute model function and Jacobian for initial parameter values
if ismember(JacStatus,'on')
    [y,J_prevTmp] = fh(theta_prev,p.Results.u);
elseif ismember(JacStatus ,'off')
    y = fh(theta_prev,p.Results.u);
    J_prevTmp = JacApprox(theta_prev,y,fh,p);
end
J_prev = Jw*J_prevTmp; % Scale Jacobian with residual weights

[cF_prev,F_prev] = costFunctionEval(y,p.Results.d,p.Results.s);

cF = cF_prev*10; % Induce that the present cost-function is worse than previous one with the assumed value of lambda
iterUpdate = 1;
deltaT = 1E6;
lambda = 10;
innerLoop = 1; % Used for de-bugging purposes

cF_iter(iterUpdate,1) = cF;
L_iter(iterUpdate,1) = lambda;

% Start solution update
while norm(deltaT) > p.Results.TolDT && iterUpdate <= p.Results.iterMax 
    if isinf(y)
        error('\nLM Model output is infinity at iteration %d\n',iterUpdate);
    end
    while cF > cF_prev % Increase lambda and re-evaluate parameter update
        lambda = lambda*10;
        deltaT = parameterUpdate(J_prev,F_prev,lambda);     % Calculate parameter update
        theta = theta_prev + deltaT;                        % Update parameter estimate
        
        % Evalaute model function and Jacobian for updated parameter
        if ismember(JacStatus ,'on')
            [y, J_Tmp] = fh(theta,p.Results.u);
        elseif ismember(JacStatus ,'off')
            y = fh(theta,p.Results.u);
            J_Tmp = JacApprox(theta,y,fh,p);
        end
        J = Jw*J_Tmp;                                        % Scale Jacobian with residual weights
        [cF, F] = costFunctionEval(y,p.Results.d,p.Results.s);
        innerLoop = innerLoop + 1;
    end
    
    cF_prev = cF;
    theta_prev = theta;
    J_prev = J;
    F_prev = F;
    
    lambda = lambda/10;
    [deltaT,regRank] = parameterUpdate(J,F,lambda);       % Calucuate parameter update
    theta = theta + deltaT;                               % Update parameter estimate
    
    % Evalaute model function and Jacobian for updated parameter
    if ismember(JacStatus ,'on')
        [y,J_Tmp] = fh(theta,p.Results.u);
    elseif ismember(JacStatus ,'off')
        y = fh(theta,p.Results.u);
        J_Tmp = JacApprox(theta,y,fh,p);
    end
    J = Jw*J_Tmp;                                        % Scale Jacobian with residual weights
    [cF, F] = costFunctionEval(y,p.Results.d,p.Results.s);
    
    cF_iter(iterUpdate,1) = cF;
    L_iter(iterUpdate,1) = lambda;    
    
    if regRank.unique == 0
        results.rankMsg{iterUpdate,1} =  sprintf(['LM Regressor rank deficient at iteration %d %s'],iterUpdate,regRank.msg);
        if strcmp(p.Results.dispMsg,'on')
            fprintf(['\nLM Regressor rank deficient at iteration %d %s'],iterUpdate,regRank.msg);
        end
        results.LMRankFull(iterUpdate,1) = regRank.unique;
    else
        results.rankMsg{iterUpdate,1} = sprintf(['LM Regressor preserves full rank at iteration %d %s'],iterUpdate,regRank.msg);
        results.LMRankFull(iterUpdate,1) = regRank.unique;
    end    
    
    iterUpdate = iterUpdate + 1;
end % End of main while iterative loop
iterUpdate = iterUpdate - 1; % Reduce iteration count by one when loop is exited

% Estiamte parameter covariance matrix
if ismember('s',p.UsingDefaults) % If measurement variance is not used in the cost function eistamte from residue for paramter variance scaling
    sCF = cF/(nData-nPara);
else
    sCF = 1;                     % Else if measurement variance is used in the cost function, paramter variance estimate does not need scaling
end
covTheta = CovTheta(sCF,J);      % Parameter variance

if ismember(lower(p.Results.diagnosis),'on')
    diagnosis(cF_iter,L_iter,iterUpdate)
end

idx = [norm(deltaT) < p.Results.TolDT, iterUpdate == p.Results.iterMax];
termStr = {[' Parameter update is smaller than specified tolerance, TolDT = ', num2str(p.Results.TolDT),'.'],...
           [' Maximum iteration reached, iterMax = ', num2str(p.Results.iterMax),'.']};
       if strcmp(p.Results.dispMsg,'on')
           fprintf('\n\nIteration terminated: %s\n',termStr{idx});
       end

results.covTheta = covTheta;
results.stdTheta = sqrt(diag(covTheta));
results.cF_iter = cF_iter;
results.L_iter = L_iter;
results.Jacobian = J;
results.termMsg = ['Iteration terminated: ',termStr{idx}];
results.fracErr = results.stdTheta./abs(theta); % Fractional error
end

function valid = checkFunctionHandle(fh)
testFH = functions(fh);
if testFH.function
    valid = true;
else
    valid = false;
end
end

function valid = checkS(v)
zeroEl = sum(v == 0);
if isnumeric(v) && zeroEl == 0 % Weights should be numeric and not zero
    valid = true;
else
    error('Weights should be numeric and nonzero')
end
end


function [cF,F] = costFunctionEval(y,d,s)
F = (y-d)./s;       % Weighted residual
cF = norm(F)^2;     % Cost-function
end


function [deltaT,regRank] = parameterUpdate(J,F,lambda)
K = ((J')*J + lambda*diag(diag((J')*J)));       % Create LM regressor matrix (cost-function Hessian + Steepest descent)
Z = (-J')*F;                                    % Negative cost-function gradient
[deltaT,regRank] = Lls(K,Z);                    % Call numerically stable linear least squares method
end


function diagnosis(cF,L,I)
figure()
semilogx([0:I-1],cF,'.-')
xlabel('Iteration number')
ylabel('Cost-fucntion (y-d/s)^2')

figure()
plot([0:I-1],L,'.-')
xlabel('Iteration number')
ylabel('Steepest descent lambda factor')
end



function [theta,regRank] = Lls(K,Z)
% Computes a numerically stable linear least squares estimate
% and returns the optimal estimate and rank message
%
% Inputs (mandatory): 
%   K: Regresor matrix, size n x m
%   Z: Output vector, size n x 1
%
% Outputs:
%   theta: Optimum parameter estimate, size m x 1
%   regRank: Structure variable with fields 'msg' and 'unique'. Message and 
%            flag if regressor matrix losses rank or not.

% Normalise K with the 1/norm of each column
Knorm = sqrt(sum(abs(K).^2));
idxZeros = Knorm<1E-14;
Knorm(idxZeros) = 1;
N = diag(1./Knorm);
Kn = K*N;

% Compute Lls via economic SVD decompostion
[U, S, V] = svd(Kn,0);          % Perform SVD
ss = diag(S);                   % Singular values
idxZeros = ss < 1E-14;          % Replace any small singular value with inf 
nCol = size(Kn,2);
if sum(idxZeros)>0              % If there are zero singular values sum(idxZeros) > 0 and regressor is rank deficient
    regRank.msg = sprintf('\nRank = %d instead of %d. \nParameter update estimated from a subspace.',nCol-sum(idxZeros),nCol);
    regRank.unique = 0;
else
    regRank.msg = sprintf('\nRank =  %d.',nCol);
    regRank.unique = 1;
end
ss(idxZeros) = inf;
Sv = diag(1./ss);  %Inverse singular value matrix

% Least squares solution
theta = N*V*Sv*(U')*Z;

end

function ct = CovTheta(s,J)
% Numerically stable calculation of the parameter covaraince matrix.
%
% Inputs
%   s: Sum of squared residuals/(nDataPts - nPara) or 1, if CF is not weighted or weighted respectively
%   J: Jacobian matrix
%
% Outputs
%   ct: Parameter covariance matrix

% Normalise J with the 1/norm of each column
Jnorm = sqrt(sum(abs(J).^2));
idxZeros = Jnorm<1E-14;
Jnorm(idxZeros) = 1;
N = diag(1./Jnorm);
Jn = J*N;

[~, S, V] = svd(Jn,0);          % Perform SVD
ss = diag(S);                   % Singular values
idxZeros = ss < 1E-14;          % Replace any small singular value with inf 
nCol = size(Jn,2);
if sum(idxZeros)>0
    regRank.msg = sprintf('\nRegressor is rank defficient. Rank = %d instead of %d. \nParameters estimated from a subspace.',nCol-sum(idxZeros),nCol);
    regRank.unique = 0;
else
    regRank.msg = sprintf('Estimated parameters are unique.\nRegressor preserves full rank. Rank =  %d.', nCol);
    regRank.unique = 1;
end

ss(idxZeros) = inf;
Sv = diag(1./ss);  %Inverse singular value matrix

ct = (N')*V*(Sv)*Sv*(V')*N*s;    % Parameter covariance matrix
end


function J = JacApprox(theta,y,fh,p)
% Approximate Jacobian with a first order finite difference
%
% Inputs:
%   theta : Paramater vector. Size nTheta x 1
%   y: Model output evaluated at theta. Size nData x 1
%   fh: Function handle
%   p: Structure to epsilon and input data to simulate model function
%
% Outputs:
%   J: Approximated Jacobian size nData x nTheta

nTheta = length(theta);
nData = length(y);
J = zeros(nData,nTheta);

epsilon = p.Results.epsilon;
deltaTheta = epsilon*theta;                     % Multiply vector of all the parameters by epsilon
for nn = 1:nTheta
    eVec = zeros(nTheta,1);
    eVec(nn) = 1;
    theta_inc = theta + deltaTheta(nn)*eVec;    % Select the increment for each parameter
    yInc = fh(theta_inc,p.Results.u);
    J(:,nn) = (yInc-y)./deltaTheta(nn);
end
end