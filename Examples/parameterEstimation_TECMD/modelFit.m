function results = modelFit(x,y)
% Compute the RMSE, Pk error and goodness of fit for a simulated (x) and
% measured signal (y)
%
% Copyright (C) W. D. Widanage -  WMG, University of Warwick, U.K. 08/11/2019 (Mama Said)
% All Rights Reserved
% Software may be used freely for non-comercial purposes only

x = x(:);
y = y(:);
error = x-y;
results.RMSE = sqrt(mean(error.^2,'omitnan'));
results.PkErr = max(abs(x-y));
results.gof = max([0, 1 - (results.RMSE^2/var(y,'omitnan'))]);
end