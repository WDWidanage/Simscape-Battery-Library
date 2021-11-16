function [thetaOpt, paraInfo, vFit] = EstECMPara(t,u,v,varargin)
% Estimate the parameters of an ECM. Postive current is assumed as
% discharge
%
% W.D. Widanage 21/12/2018

p = inputParser; % Create an input parse object to handle positional and property-value arguments


% Create variable names and assign default values after checking the value
addRequired(p,'t', @isnumeric);
addRequired(p,'u', @isnumeric);
addRequired(p,'v', @isnumeric);

% Optional parameters
addParameter(p,'Jacobian','on')
addParameter(p,'order',1)
addParameter(p,'plotFit',0,@isnumeric)
addParameter(p,'ecmFitSeriesCap','off') % Set this to on to fit a ECM with a series capacitor
addParameter(p,'dispMsg','on')
% options = optimset('TolFun',1e-5,'MaxFunEvals',2000,'Jacobian','on','Algorithm','trust-region-reflective');

% Re-parse parObj
parse(p,t, u, v, varargin{:})

t = p.Results.t;
u = p.Results.u;
v = p.Results.v;

order = p.Results.order;
plotFit = p.Results.plotFit;

% Ensure inputs are column vectors
t = t(:);
u = u(:);
v = v(:);


% De-trend OCV from the voltage
ocvStart = v(1); tStart = t(1);
ocvEnd = v(end); tEnd = t(end);
ocvRef = interp1([tStart;tEnd],[ocvStart;ocvEnd],t);
volOCVRemoved = v - ocvRef + ocvStart;

OCV = v(1);

% Fit equivalent circuit with series capacitior on time data
if ismember(p.Results.ecmFitSeriesCap,{'on','On'})
    dtTmp = diff(t);
    dt = [dtTmp(1);dtTmp];
    fh = @(theta,u)EquivalentCircuitModel(theta,u,OCV,dt,order);    % Model function handle
    % Set initial values for ECM parameters
    [~, idx] = max(abs(u));
    if sign(u(idx)) < 0                                 % For a charge pulse
        Ro = (max(v) - OCV)/abs(u(idx));
        OCVacc0 = (v(end) - OCV)/(abs(u(idx))*10); % OCV', Steady state voltage = OCVacc*Area under applied current + OCV
    else                                                % For a discharge pulse
        Ro = (OCV - min(v))/abs(u(idx));
        OCVacc0 = (OCV - v(end))/(abs(u(idx))*10); % OCV', Steady state voltage = OCVacc*Area under applied current + OCV
    end
    
    for mm = 1: order
        Rp(mm) = 1e-3;
        tau(mm) = 10^(mm-1);
    end
    
    theta0 = [Ro OCVacc0 Rp tau]';
    options.Jacobian = p.Results.Jacobian;
    options.dispMsg = p.Results.dispMsg;
    [thetaOpt,paraInfo] = LMAlgorithm(fh, theta0, u, v, options);
    %     [thetaOpt,resNorm] = lsqcurvefit(fh,theta0,u,volOCVRemoved);
    
    % Simulate ECM with estimation data set and estimated optimum parameteres
    vFit = EquivalentCircuitModel(thetaOpt,u,OCV,dt,order);
else
    % Set initial values for ECM parameters
    [~, idx] = max(abs(u));
    if sign(u(idx)) < 0                                 % For a charge pulse
        Ro = (max(v) - OCV)/abs(u(idx));
    else                                                % For a discharge pulse
        Ro = (OCV - min(v))/abs(u(idx));
    end
    
    for mm = 1: order
        Rp(mm) = 1e-3;
        tau(mm) = 5*(mm);
    end
    
    theta0 = [Ro Rp tau]';
    fh = @(theta,u)ECMStable(theta, u, OCV, t, order);    % Model function handle
    options.Jacobian = p.Results.Jacobian;
    options.dispMsg = p.Results.dispMsg;
    [thetaOpt,paraInfo] = LMAlgorithm(fh, theta0, u, volOCVRemoved, options);
    %     [thetaOpt,resNorm] = lsqcurvefit(fh,theta0,u,volOCVRemoved);
    
    yECMTmp = ECMStable(thetaOpt, u, OCV, t, order);
    vFit = yECMTmp + ocvRef - ocvStart;
end



errorECM = v - vFit;
paraInfo.resNorm = sqrt(mean(errorECM.^2));
paraInfo.pkErrECM = max(abs(errorECM));

if plotFit == 1
    figure();
    subplot(2,1,1)
    plot(t,u,'. -');
    ylabel('Current (A)');
    subplot(2,1,2)
    plot(t,v,'- .',t,vFit,'-')
    xlabel('Time (s)'); ylabel('Voltage (V)'); legend('Measured','ECM sim')
    title(['RMSE ECM: ',num2str(paraInfo.resNorm),' Pk Err ECM: ', num2str(paraInfo.pkErrECM)]);
end



function [Vl, J]= EquivalentCircuitModel(theta, Il, OCV, del, order)
%
%
% Parameters are arranged as:
%      theta = Ro, OCV', Rp1,..,Rpn, tau1,...,taun
%
% W. D. Widanage 19/10/2013 (Regain)
%

% Change parameter vector to a column vector
theta = theta(:);

% Extract parameters
Ro = theta(1);
OCVacc = theta(2);
RpAll = theta(3:2+order);
tauAll = theta(3+order:2+2*order);

dataPts = length(Il);       % Number of data points

% Initialise
Ip = zeros(order,1);
Vl = zeros(dataPts,1);
Integral = 0;

% Initialise for Jacobian
J = zeros(dataPts,length(theta));
dIpdtau = zeros(order,1);

Vl(1) = OCV - Ro*Il(1) - OCVacc*Integral - RpAll'*Ip;
J(1,:) = [-Il(1), -Integral, -Ip', -dIpdtau'];

for ii = 2:dataPts
    Ts = del(ii-1);                                    % Sampling interval
    expTau = exp(-Ts./tauAll);
    for jj = 1:order
        Ip(jj,1) = expTau(jj)*(Ip(jj,1) - Il(ii)) + Il(ii);
    end
    Integral = Integral + (Il(ii) + Il(ii-1))*Ts/2;
    
    Vl(ii) = OCV - Ro*Il(ii) - OCVacc*Integral - RpAll'*Ip;
    
    % Create Jacobian matrix
    for jj = 1:order
        dIpdtau(jj,1) = expTau(jj)*(dIpdtau(jj,1) + Ip(jj,1)*Ts/tauAll(jj)^2 - Il(ii)*Ts/tauAll(jj)^2);
    end
    RdIpdtau = RpAll.*dIpdtau;
    J(ii,:) = [-Il(ii), -Integral, -Ip', -RdIpdtau'];
end

function [Vl, J] = ECMStable(theta, Il, OCV, timeVec, order)
% Stable equivalent cirucit model, without series capacitor
%
% Parameters are arranged as:
%      theta = Ro, Rp1,..,Rpn, tau1,...,taun
%
% W. D. Widanage 04/07/2014 (Serene)
%
dataPts = length(Il);       % Number of data points

% Check if weighting function is provided. else initialise to ones

% Change parameter vector to a column vector
theta = theta(:);

% Extract parameters
Ro = theta(1);
RpAll = theta(2:1+order);
tauAll = theta(2+order:1+2*order);



% Initialise
Ip = zeros(order,1);
Vl = zeros(dataPts,1);

% Initialise for Jacobian
J = zeros(dataPts,length(theta));
dIpdtau = zeros(order,1);

Vl(1) = OCV - Ro*Il(1) - RpAll'*Ip;
J(1,:) = [-Il(1), -Ip', -dIpdtau'];

for ii = 2:dataPts
    Ts = timeVec(ii)-timeVec(ii-1);                                    % Sampling interval
    expTau = exp(-Ts./tauAll);
    for jj = 1:order
        Ip(jj,1) = expTau(jj)*(Ip(jj,1) - Il(ii)) + Il(ii);
    end
    
    Vl(ii) = OCV - Ro*Il(ii) - RpAll'*Ip;
    
    % Create Jacobian matrix
    for jj = 1:order
        dIpdtau(jj,1) = expTau(jj)*(dIpdtau(jj,1) + Ip(jj,1)*Ts/tauAll(jj)^2 - Il(ii)*Ts/tauAll(jj)^2);
    end
    RdIpdtau = RpAll.*dIpdtau;
    J(ii,:) = [-Il(ii), -Ip', -RdIpdtau'];
    
    
end

Vl = Vl(:); % Change to a column vector
