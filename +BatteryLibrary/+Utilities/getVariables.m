function  [dv,SF] = getVariables(simout, block_names, nvp)
% Get the distibuted, average and time variables of TECMD, TSPMe and TSPMeA
% via post processing
%
% Mandtory inputs
%   simout: This is the output that Simulink generates in the workspage
%           when you run a Simulink model or when you call the "sim"
%           command to simulate a Simulink model.
%   block_names: Specify the names of the BatteryLibrary components that
%               were used in the simulink model as a vector of strings.
%
% W.D.Widanage 24/05/2023 (Nothing else matters)

arguments
    simout Simulink.SimulationOutput
    block_names {mustBeText}
    nvp.state_name = "xout";
    nvp.time_limits = [];        % Specify the time limits [s] to extract the data
end

import BatteryLibrary.Utilities.*

try
    simout.(char(nvp.state_name));
    state_name = nvp.state_name;
catch
    error(sprintf("The states are not saved in the current Simulink model. \nPlease go to the Siumlink ""Model Settings"" and in the ""Data Import/Export"" section enable the ""States"" check box option in ""Save to workspace"" and re-run model" ))
end

import BatteryLibrary.Utilities.*
num_of_states = simout.(state_name).numElements;
simout_states = simout.(state_name);

for bb = 1:num_of_states
    simout_block_path_list_tmp(bb) = string(simout_states{bb}.BlockPath.convertToCell);
    simout_block_names_tmp = split(simout_block_path_list_tmp(bb),"/");
    simout_block_name_list(bb) = simout_block_names_tmp(end);
end

% Get unique block names and paths from the simout object
[simout_block_names,idx] = unique(simout_block_name_list);
simout_block_path_list = simout_block_path_list_tmp(idx);

for bb = 1:numel(block_names)

    block_name = block_names(bb);

    if  ismember(block_name,simout_block_names)
        [~,blk_idx] = ismember(block_name,simout_block_names);
        block_path = simout_block_path_list(blk_idx);
        model_para = getModelParameters(block_path);    % Get dialog parameters of each of the desired block_names

        % Get the type of the simscape battery block: TECMD, TSPMe or TSMPeA
        model_type = model_para.(block_name).model_type;

        % Get time range
        sim_time = simout.tout;
        if isempty(nvp.time_limits)
            time_limits = [sim_time(1),sim_time(end)];
            idx_rng = 1:numel(sim_time);
        else
            time_limits = nvp.time_limits;
            idx_rng = find(sim_time >= time_limits(1),1,"first") : find(sim_time <= time_limits(2),1,"last");
        end

        if ismember(model_type, "TECM")
            T = getSignal(simout_states,block_path,"T",time_limits);
            SF = scalingFactors(model_para.(block_name),T);
            i = getCurrent(simout_states,block_path,model_para.(block_name),SF,time_limits);
            z = getSignal(simout_states,block_path,"z",time_limits);
            V = getVoltage(simout_states,block_path,model_para.(block_name),z,time_limits);

            dv.(block_name).T.T = T;
            dv.(block_name).I = i;
            dv.(block_name).V = V;
            dv.(block_name).SoC.z = z;
            dv.(block_name).model_type = model_type;
            dv.(block_name).block_present = 1;
            dv.(block_name).time = sim_time(idx_rng);
        end

        if ismember(model_type,"TECMD")
            T = getSignal(simout_states,block_path,"T",time_limits);
            SF = scalingFactors(model_para.(block_name),T);
            i = getCurrent(simout_states,block_path,model_para.(block_name),SF,time_limits);
            z = getSoC(simout_states,block_path,model_para.(block_name),time_limits);
            V = getVoltage(simout_states,block_path,model_para.(block_name),z,time_limits);

            dv.(block_name).x.x = model_para.(block_name).xn;
            dv.(block_name).I = i;
            dv.(block_name).V = V;
            dv.(block_name).T.T = T;
            dv.(block_name).SoC = z;
            dv.(block_name).model_type = model_type;
            dv.(block_name).block_present = 1;
            dv.(block_name).time = sim_time(idx_rng);

        end

        if ismember(model_type,["TSPMe","TSPMeA"])
            r = getParticleDims(model_para.(block_name));
            x = getElectrodeSeperatorDims(model_para.(block_name));
            c = getConcentrations(simout_states,block_path,model_para.(block_name),time_limits);
            T = getSignal(simout_states,block_path,"T",time_limits);
            SF = scalingFactors(model_para.(block_name),T);
            i = getCurrent(simout_states,block_path,model_para.(block_name),SF,time_limits);
            z = getSoC(simout_states,block_path,model_para.(block_name),time_limits);
            [phi,V] = getPotentials(simout_states,block_path,model_para.(block_name),SF,i,time_limits);

            dv.(block_name).c = c;
            dv.(block_name).r = r;
            dv.(block_name).x = x;

            dv.(block_name).T.T = T;
            dv.(block_name).I = i;
            dv.(block_name).Phi = phi;
            dv.(block_name).V = V;
            dv.(block_name).SoC = z;
            dv.(block_name).model_type = model_type;
            dv.(block_name).block_present = 1;
            dv.(block_name).time = sim_time(idx_rng);
        end

        if ismember(model_type,"TSPMeA")
            dv.(block_name).SR = getSideReactions(simout_states,block_path,model_para.(block_name),SF,i,time_limits,phi);
        end


    else
        simout_block_names_tmp = split(simout_block_path_list(bb),"/");
        simulink_filename = simout_block_names_tmp(1);
        warning(sprintf("A block named %s is not present in the simulink model %s.slx",block_names(bb),simulink_filename))
        dv.simulink_filename = simulink_filename;
        dv.(block_name).block_present = 0;
        SF = [];
    end


end

end

function r = getParticleDims(para)

arguments
    para
end
% Get particle radiusand convert to SI units
Rn_SI = value(simscape.Value(para.Rn,para.Rn_unit),'m');
Rp_SI = value(simscape.Value(para.Rp,para.Rp_unit),'m');

% Get collocation paricle collocation points
s10 = para.s10;

% Dimensional particle radius [m]
r.n = (1+s10)/2*Rn_SI; % Negative electrode radial domain collocation points on interval [0,Rn]
r.p = (1+s10)/2*Rp_SI; % Positve electrode radial domain collocation points on interval [0,Rp]
end

function x = getElectrodeSeperatorDims(para)
arguments
    para
end

% Get electrode and seperator dimensions and convert to SI units
Ln_SI = value(simscape.Value(para.Ln,para.Ln_unit),'m');
Ls_SI = value(simscape.Value(para.Ls,para.Ls_unit),'m');
Lp_SI = value(simscape.Value(para.Lp,para.Lp_unit),'m');

L_SI = Ln_SI + Ls_SI + Lp_SI; % Total electrode thickness
lp = Lp_SI/L_SI;
ls = Ls_SI/L_SI;
ln = Ln_SI/L_SI;

% Get collocation paricle collocation points
s6 = para.s6;
s10 = para.s10;

xn = ln*(1+s10)/2;           % -ve electrode domain collocation points on interval [0,ln]
xs = ln + ls*(1+s6)/2;       % Separator domain collocation points on interval [ln,ln+ls]
xp = (ln+ls) + lp*(1+s10)/2; % +ve electrode domain collocation points on interval [ln+ls,lp]

% Dimensional electrode lengths [m]
x.n = xn*L_SI;                      % Negative electrode domain [0, Ln]
x.p = xp*L_SI;                      % Positive electrode domain [Ln+Ls, Lp]
x.s = xs*L_SI;
x.x = [xn,xs(2:end-1),xp]*L_SI;     % Full electrode domain [0,L]
x.L = L_SI;                         % Electrod length [m]

% Save non-dimensional quantitites
x.lp = lp;
x.ls = ls;
x.ln = ln;
end

function c = getConcentrations(states,block_path,para,time_limits)

arguments
    states,
    block_path
    para
    time_limits
end

% Get postive electrode dimenstions
x = getElectrodeSeperatorDims(para);
dx = diff(x.p/x.L);

cn_max = value(simscape.Value(para.cn_max,para.cn_max_unit),'mol/m^3');
cp_max = value(simscape.Value(para.cp_max,para.cp_max_unit),'mol/m^3');
ce0 = value(simscape.Value(para.ce0,para.ce0_unit),'mol/m^3');

cn_hat = getSignal(states,block_path,"cn_hat",time_limits);
cp_hat = getSignal(states,block_path,"cp_hat",time_limits);
cp_hat_bar = 0.5*(cp_hat(:,1:end-1) + cp_hat(:,2:end))*dx'/x.lp;

cen_hat = getSignal(states,block_path,"cen_hat",time_limits);
ces_hati = getSignal(states,block_path,"ces_hati",time_limits);
cep_hat = getSignal(states,block_path,"cep_hat",time_limits);

% Non-dimensional
c.cp_hat_bar = cp_hat_bar;
c.sp = cp_hat(:,end);
c.cen_hat = cen_hat;
c.ces_hati = ces_hati;
c.cep_hat = cep_hat;


% Dimensional
c.cn = cn_hat*cn_max;
c.cp = cp_hat*cp_max;
c.ce = [cen_hat, ces_hati, cep_hat]*ce0;
end

function  i = getCurrent(states,block_path,para,sf,time_limits)

arguments
    states,
    block_path,
    para,
    sf
    time_limits
end

if ismember(para.model_type,["TSPMe","TSPMeA"])
    % Get cp stiochiometry and and cep_hat
    c = getConcentrations(states,block_path,para,time_limits);

    % Get postive electrode dimenstions
    x = getElectrodeSeperatorDims(para);
    Area_p = value(simscape.Value(para.Area_p,para.Area_p_unit),'m^2');

    etaBV_p = getSignal(states,block_path,"etaBV_p",time_limits);

    i0p = sf.gamma_p./sf.CC_rp.*sqrt(c.cep_hat(:,end).*c.sp.*(1-c.sp));

    i.i_hat = sinh(etaBV_p(:,end))*x.lp.*i0p;   % Non-dimensional current density, positive is discharing in model equations
    i.I = -i.i_hat*Area_p*sf.i0;                % Dimensional current [A], positive is charging in the simscape format

elseif ismember(para.model_type,"TECMD")
    etaO = getSignal(states,block_path,"etaO",time_limits);
    i.I = etaO./sf.Ro;

elseif ismember(para.model_type,"TECM")
    try
        i.I = getSignal(states,block_path,"i",time_limits);
    catch
        v = getSignal(states,block_path,"v",time_limits);
        U = getSignal(states,block_path,"OCV",time_limits);
        etaP = getSignal(states,block_path,"etaP",time_limits);
        Ro = getSignal(states,block_path,"Ro",time_limits);

        etaO = v - U - etaP;
        i.I = etaO./Ro;
    end
end

end

function [phi,V] = getPotentials(states,block_path,para,sf,i,time_limits)

x = getElectrodeSeperatorDims(para);
Tref = value(simscape.Value(para.Tref,para.Tref_unit),'K');
tr = para.tr;

F = 96485.3329;   % Faraday constant
R = 8.3144;       % Universal gas constant

xn = x.n/x.L;           % Non-dimensional negative electrode points
xs = x.s/x.L;           % Non-dimensional seperator electrode points
xp = x.p/x.L;           % Non-dimensional positive electrode points

ln = x.ln;
lp = x.lp;

nE = para.nE;
nS = para.nS;

M6 = tril(ones(nS));    % Lower triangular matrix to perform cumulative integration
M10 = tril(ones(nE));   % Lower triangular matrix to perform cumulative integration

dxn = diff(xn);
dxs = diff(xs);
dxp = diff(xp);

try
    eps_n = getSignal(states,block_path,"eps_n",time_limits);
catch
    eps_n = para.eps_n;
end
eps_s = para.eps_s;
eps_p = para.eps_p;

cen_hat = getSignal(states,block_path,"cen_hat",time_limits);
ces_hati = getSignal(states,block_path,"ces_hati",time_limits);
cep_hat = getSignal(states,block_path,"cep_hat",time_limits);
etaBV_n = getSignal(states,block_path,"etaBV_n",time_limits);
etaBV_p = getSignal(states,block_path,"etaBV_p",time_limits);
cn_hat = getSignal(states,block_path,"cn_hat",time_limits);
cp_hat = getSignal(states,block_path,"cp_hat",time_limits);
ces_hat = [cen_hat(:,nE), ces_hati, cep_hat(:,1)];                             % Enforce separator boundary electrolyte identity

J_n = i.i_hat/x.ln;
J_p = -i.i_hat/x.lp;
Sigma_e = sf.Sigma_e;
Sigma_n = sf.Sigma_n;
Sigma_p = sf.Sigma_p;
gamma_T = sf.gamma_T;
lambda = sf.lambda;
beta_sr = sf.beta_sr;
Sigma_film_n = sf.Sigma_film_n;
epsn_init = para.eps_n0;
Lfn_hat = 1 - (eps_n - epsn_init)/beta_sr;                                    % Negative electrode film thickness (Ref#3 eq C.8)

% OCVs
sn = cn_hat(:,end);
sp = cp_hat(:,end);
Up = -0.809*sp + 4.4875 - 0.0428*tanh(18.5138*(sp-0.5542)) - 17.7326*tanh(15.7809*(sp-0.3117))+17.5842*tanh(15.9308*(sp-0.3120));
Un = 1.9793*exp(-39.3631*sn)+0.2482-0.0909*tanh(29.8538*(sn-0.1234))- 0.04478*tanh(14.9159*(sn-0.2769)) - 0.0205*tanh(30.44*(sn-0.6103));

That = sf.That;

num_time_points = size(Sigma_e,1);

% Initialise
Phien_hat = nan(num_time_points,nE);
Phies_hat = nan(num_time_points,nS);
Phiep_hat = nan(num_time_points,nE);
Phin_hat = nan(num_time_points,nE);
Phip_hat = nan(num_time_points,nE);

for tt = 1:num_time_points
    ss_en = sf.sigma_e(tt)*ones(nE,1)/para.sigma_eRef;
    ss_es = sf.sigma_e(tt)*ones(nS,1)/para.sigma_eRef;
    ss_ep = sf.sigma_e(tt)*ones(nE,1)/para.sigma_eRef;


    % Electrolyte potential terms
    inG_n = J_n(tt)*xn'./(Sigma_e(tt)*ss_en.*(eps_n(tt,:)'.^1.5));   % Integrand term in Ref#1 eq B.18 ##
    inG_s = i.i_hat(tt)./(Sigma_e(tt)*ss_es*eps_s^1.5);             % Integrand term in Ref#1 eq B.18
    inG_p = J_p(tt)*(1-xp)'./(Sigma_e(tt)*ss_ep*eps_p^1.5);             % Integrand term in Ref#1 eq B.18

    int_n = M10*[0;0.5*dxn'.*(inG_n(1:nE-1)+inG_n(2:nE))];          % Cumulative trapezoidal integration
    int_s = M6*[int_n(nE);0.5*dxs'.*(inG_s(1:nS-1)+inG_s(2:nS))];   % Cumulative trapezoidal integration
    int_p = M10*[int_s(nS);0.5*dxp'.*(inG_p(1:nE-1)+inG_p(2:nE))];  % Cumulative trapezoidal integration

    Phien_hat(tt,:) = -int_n' + 2*(1-tr)*log(cen_hat(tt,:)/cen_hat(tt,1));          % Ref#1 eq B.18
    Phies_hat(tt,:) = -int_s' + 2*(1-tr)*log(ces_hat(tt,:)/cen_hat(tt,1));          % Ref#1 eq B.18
    Phiep_hat(tt,:) = -int_p' + 2*(1-tr)*log(cep_hat(tt,:)/cen_hat(tt,1));          % Ref#1 eq B.18


    % Phin_hat calculation terms (Ref#1 eq B.27a and Ref#3 eq C.9)
    iappTerm_n = -i.i_hat(tt)*(2*ln - xn').*xn'/(2*ln*Sigma_n(tt)) + i.i_hat(tt)*ln/(3*Sigma_n(tt));                % First two terms of Ref#1 eq B.27a
    intPhin_1 = -1/ln*0.5*dxn*(int_n(1:nE-1) + int_n(2:nE));                                        % Third term of Ref#1 eq B.27a
    intG_logCn = log(cen_hat(tt,:)/cen_hat(tt,1))';                                                  % Integrand in fourth term of Ref#1 eq B.27a
    intPhin_2 = 2*(1-tr)*(1+gamma_T*That(tt))*0.5*dxn*(intG_logCn(1:nE-1) + intG_logCn(2:nE))/ln;   % Fourth term of Ref#1 eq B.27a
    intG_BVn = etaBV_n(tt,:)';                                                                             % Integrand in fifth term of Ref#1 eq B.27a
    intPhin_3 = 2/ln*(1+gamma_T*That(tt))*0.5*dxn*(intG_BVn(1:nE-1) + intG_BVn(2:nE));                  % Fifth term of Ref#1 eq B.27a
    Lfn_hat_bar = 0.5*dxn*(Lfn_hat(tt,1:nE-1) + Lfn_hat(tt,2:nE))'/ln;                    % Average film thickness (trapezoidal integration over spatial coordinates)
    iappFilm = i.i_hat(tt)*Lfn_hat_bar/(ln*Sigma_film_n(tt));                                                 % Fourth term of Ref#3 eq C.9
    Phin_hat(tt,:) = Un(tt) + (iappTerm_n + intPhin_1 + intPhin_2 + intPhin_3 + 0*iappFilm)/lambda;             % Negative electrode potential (Ref#1 eq B.28a), with ageing potential drop added (from Ref#3 eq C.9)


    % Phip_hat calculation terms (Ref#1 eq B.27b)
    iappTerm_p = i.i_hat(tt)*(2*(1-lp) - xp').*xp'/(2*lp*Sigma_p(tt)) - i.i_hat(tt)*(2*lp^2 - 6*lp + 3)/(6*lp*Sigma_p(tt)); % First two terms of Ref#1 eq B.27b
    intPhip_1 = -1/lp*0.5*dxp*(int_p(1:nE-1) + int_p(2:nE));                                            % Third term of Ref#1 eq B.27b
    intG_logCp = log(cep_hat(tt,:)/cen_hat(tt,1))';                                                               % Integrand in fourth term of Ref#1 eq B.27b
    intPhip_2 = 2*(1-tr)*(1+gamma_T*That(tt))*0.5*dxp*(intG_logCp(1:nE-1) + intG_logCp(2:nE))/lp;           % Fourth term of Ref#1 eq B.27a
    intG_BVp = etaBV_p(tt,:)';                                                                                 % Integrand in fifth term of Ref#1 eq B.27b
    intPhip_3 = -2/lp*(1+gamma_T*That(tt))*0.5*dxp*(intG_BVp(1:nE-1) + intG_BVp(2:nE));                     % Fifth term of Ref#1 eq B.27b
    Phip_hat(tt,:) = Up(tt) + (iappTerm_p + intPhip_1 + intPhip_2 + intPhip_3)/lambda;                             % Positive electrode potential (Ref#1 eq B.28b)

end

phi.Phie_hat = [Phien_hat,Phies_hat(:,2:nS-1),Phiep_hat];
phi.Phien_hat = Phien_hat;
phi.Phie = phi.Phie_hat*R*Tref/F;
phi.Phin = Phin_hat;                        % Both non-dimensional and diemnsional are the same
phi.Phip = Phip_hat;                        % Both non-dimensional and diemnsional are the same
phi.Un = Un;
phi.Up = Up;
phi.etaBV_n = etaBV_n;
phi.etaBV_p = etaBV_p;
V.V = Phip_hat(:,end) - Phin_hat(:,1);
V.U = Up - Un;

end

function z = getSoC(states,block_path,para,time_limits)

arguments
    states,
    block_path,
    para,
    time_limits
end

if ismember(para.model_type,["TSPMe","TSPMeA"])
    % Get positive particle concentrations
    c = getConcentrations(states,block_path,para,time_limits);

    z.z_bar = (para.sp_max - c.cp_hat_bar)/(para.sp_max - para.sp_min);   % Average SoC over postive particle
    z.z = (para.sp_max - c.sp)/(para.sp_max - para.sp_min);               % SoC at particle boundary

elseif ismember(para.model_type,"TECMD")
    z_dist = getSignal(states,block_path,"z",time_limits);
    dx = diff(para.xn);

    z.z_dist = z_dist;                                       % Distributed SoC over x
    z.z = z_dist(:,end);                                          % SoC at particle boundary
    z.z_bar = 0.5*(z_dist(:,1:end-1) + z_dist(:,2:end))*dx'; % Average SoC over x - trapezoidal integration

end

end

function SR = getSideReactions(states,block_path,para,sf,i,time_limits,phi)

F = 96485.3329;   % Faraday constant
eps_n0 = para.eps_n0;
nE = para.nE;
ULi_hat = para.ULi;
Ln = value(simscape.Value(para.Ln,para.Ln_unit),'m');
Lf0 = value(simscape.Value(para.Lf0,para.Lf0_unit),'m');
Cn = value(simscape.Value(para.Cn,para.Cn_unit),'A*s');
Area_p = value(simscape.Value(para.Area_p,para.Area_p_unit),'m^2');
pho_sei = value(simscape.Value(para.pho_sei,para.pho_sei_unit),'kg/m^3');
pho_Li = value(simscape.Value(para.pho_Li,para.pho_Li_unit),'kg/m^3');
Msei = value(simscape.Value(para.Msei,para.Msei_unit),'kg/mol');
MLi = value(simscape.Value(para.MLi,para.MLi_unit),'kg/mol');

beta_sr = sf.beta_sr;
gamma_T = sf.gamma_T;
That = sf.That;
lambda = sf.lambda;
CC_rLi = sf.CC_rLi;
Sigma_film_n = sf.Sigma_film_n;
i0 = sf.i0;

eps_n = getSignal(states,block_path,"eps_n",time_limits);
eps_nsei = getSignal(states,block_path,"eps_nsei",time_limits);
eps_nLi = getSignal(states,block_path,"eps_nLi",time_limits);

cen_hat = getSignal(states,block_path,"cen_hat",time_limits);

x = getElectrodeSeperatorDims(para);

Phin_hat = phi.Phin;
Phien_hat = phi.Phien_hat;

L = x.L;              % Dimensional electrode length
ln = x.ln;            % Non-dimensional negative electrode length
xn = x.n/L;           % Non-dimensional negative electrode points
dxn = diff(xn);

epsn_init = eps_n0*ones(nE,1);
num_time_points = length(Sigma_film_n);

% Initialise
JLi_hat = nan(num_time_points,nE);
epsn_bar = nan(num_time_points,1);
eps_barsei = nan(num_time_points,1);
eps_barLi = nan(num_time_points,1);
Lfn_hat_bar = nan(num_time_points,1);

for tt = 1:num_time_points
    epsn_bar(tt,1) = L*0.5*dxn*(eps_n(tt,1:nE-1) + eps_n(tt,2:nE))'/Ln;
    Lfn_hat = 1 - (eps_n(tt,:)' - epsn_init)/beta_sr;                                    % Negative electrode film thickness (Ref#3 eq C.8)
    J_n = i.i_hat(tt)/ln;
    Phi_fn = J_n*Lfn_hat/Sigma_film_n(tt);                                            % Potential drop across negative electrode film (last term inside exponential of Ref#3 eq C.3b) ##
    etaLi_hat = lambda*(Phin_hat(tt,:)' - ULi_hat) - Phien_hat(tt,:)' - Phi_fn;                 % Li plating over-potential (Ref#3 eq C.3b) ##
    i0Li_hat = cen_hat(tt,:)'/CC_rLi(tt);                                                    % Li plating exchange current density (Ref#3 eq B.3)
    JLi_hat(tt,:) = -i0Li_hat.*exp(-0.1*etaLi_hat/(1+gamma_T*That(tt)));                    % Li-plaitng side reaction current (Ref#3 eq C.3b, j_li substituted with Ref#3 B.3 and thermal coupling from Ref#1)
    Lfn_hat_bar(tt,1) = 0.5*dxn*(Lfn_hat(1:nE-1) + Lfn_hat(2:nE))/ln;

    eps_barsei(tt,1) = -L*0.5*dxn*(eps_nsei(tt,1:nE-1) + eps_nsei(tt,2:nE))';
    eps_barLi(tt,1) = -L*0.5*dxn*(eps_nLi(tt,1:nE-1) + eps_nLi(tt,2:nE))';
end

SoH_sei = (F*Area_p*2*eps_barsei*pho_sei/Msei)/Cn;                            % Capacity SoH [%] from SEI  (Ref#3 eq 21 for Nsr and normalising by Cn)
SoH_Li = (F*Area_p*eps_barLi*pho_Li/MLi)/Cn;                                  % Capacity SoH [%] from Li-plating (Ref#3 eq 21 for Nsr and normalising by Cn)
SoH_tot = 1 - (SoH_sei + SoH_Li);

SR.Jsei_hat = getSignal(states,block_path,"Jsei_hat",time_limits);
SR.JLi_hat = JLi_hat;
SR.Jsei = SR.Jsei_hat*i0/L;
SR.JLi = JLi_hat*i0/L;
SR.eps_n = eps_n;
SR.eps_nsei = eps_nsei;
SR.eps_nLi = eps_nLi;
SR.Lfn = Lfn_hat*Lf0;
SR.Lfn_bar = Lfn_hat_bar*Lf0;
SR.epsn_bar = epsn_bar;
SR.SoH = SoH_tot;
SR.SoH_sei = SoH_sei;
SR.SoH_Li = SoH_Li;


end

function sf = scalingFactors(para,T)

arguments
    para
    T
end

if ismember(para.model_type,["TSPMe","TSPMeA"])
    F = 96485.3329;   % Faraday constant
    R = 8.3144;       % Universal gas constant

    x = getElectrodeSeperatorDims(para);

    % Thermal parameters
    Tref = value(simscape.Value(para.Tref,para.Tref_unit),'K');
    Ea_kp = value(simscape.Value(para.Ea_kp,para.Ea_kp_unit),'J/mol');
    kpRef = value(simscape.Value(para.kpRef,para.kpRef_unit),'A*m^(5/2)/mol^(3/2)');
    Ea_sigman = value(simscape.Value(para.Ea_sigman,para.Ea_sigman_unit),'J/mol');
    Ea_sigmae = value(simscape.Value(para.Ea_sigmae,para.Ea_sigmae_unit),'J/mol');
    Ea_sigmap = value(simscape.Value(para.Ea_sigmap,para.Ea_sigmap_unit),'J/mol');
    Ea_sigmasei = value(simscape.Value(para.Ea_sigmasei,para.Ea_sigmasei_unit),'J/mol');
    Ea_ksei = value(simscape.Value(para.Ea_ksei,para.Ea_ksei_unit),'J/mol');
    Ea_kLi = value(simscape.Value(para.Ea_kLi,para.Ea_kLi_unit),'J/mol');
    Ea_Dsei = value(simscape.Value(para.Ea_Dsei,para.Ea_Dsei_unit),'J/mol');
    sigma_nRef = value(simscape.Value(para.sigma_nRef,para.sigma_nRef_unit),'S/m');
    sigma_eRef = value(simscape.Value(para.sigma_eRef,para.sigma_eRef_unit),'S/m');
    sigma_pRef = value(simscape.Value(para.sigma_pRef,para.sigma_pRef_unit),'S/m');
    sigma_seiRef = value(simscape.Value(para.sigma_seiRef,para.sigma_seiRef_unit),'S/m');
    kseiRef = value(simscape.Value(para.kseiRef,para.kseiRef_unit),'m/s');
    kLiRef = value(simscape.Value(para.kLiRef,para.kLiRef_unit),'m/s');
    DseiRef = value(simscape.Value(para.DseiRef,para.DseiRef_unit),'m^2/s');

    Lf0 = value(simscape.Value(para.Lf0,para.Lf0_unit),'m');

    kp = kpRef*exp(Ea_kp/R*(1/Tref-1./T));                      % Arrhenius equation for postive electrode reaction rate: kp
    sigma_n = sigma_nRef*exp(Ea_sigman/R*(1/Tref-1./T));        % Arrhenius equation for negative electrode conductivity: sigma_n
    sigma_e = sigma_eRef*exp(Ea_sigmae/R*(1/Tref-1./T));        % Arrhenius equation for electrolyte conductivity: sigma_e
    sigma_p = sigma_pRef*exp(Ea_sigmap/R*(1/Tref-1./T));        % Arrhenius equation for positive electrode conductivity: sigma_p
    sigma_sei = sigma_seiRef*exp(Ea_sigmasei/R*(1/Tref-1./T));  % Arrhenius equation for sei conductivity: sigma_sei ##
    ksei = kseiRef*exp(Ea_ksei/R*(1/Tref-1./T));                % Arrhenius equation for negative electrode sei reaction rate: ksei ##
    kLi = kLiRef*exp(Ea_kLi/R*(1/Tref-1./T));                   % Arrhenius equation for negative electrode Li plating reaction rate: kLi ##
    Dsei = DseiRef*exp(Ea_Dsei/R*(1/Tref-1./T));                % Arrhenius equation for negative electrode sei diffusion coeffcient: Dsei ##


    cp_max = para.cp_max;
    cn_max = para.cn_max;
    an = para.an;
    csei0 = para.csei0;

    Cn = value(simscape.Value(para.Cn,para.Cn_unit),'A*hr');
    Area_p = value(simscape.Value(para.Area_p,para.Area_p_unit),'m^2');
    ap = value(simscape.Value(para.ap,para.ap_unit),'1/m');
    Cp = value(simscape.Value(para.Cp,para.Cp_unit),'J/K');

    L = x.x(end);
    Phi0 = 1;
    i0 = Cn/Area_p;
    t0 = F*cn_max*L/i0;
    lambda = Phi0*F/(R*Tref);       % Ratio of eletrode voltage to thermal voltage

    ce0 = value(simscape.Value(para.ce0,para.ce0_unit),'mol/m^3');

    % Non-dimensional ageing parameters
    CC_rsei = cn_max./(ksei*an*t0*csei0);                % SEI exchange current parameter (Ref#3 eq B.4) ##
    CC_sei = Lf0*cn_max./(Dsei*an*t0*csei0);             % SEI concentration parameter (Ref#3 eq eq B.4) ##
    CC_rLi = cn_max./(kLi*an*t0*ce0);                    % Li plating exchange current parameter (Ref#3 eq eq B.4) ##
    Sigma_film_n = R*Tref*sigma_sei*an*L/(F*Lf0*i0);    % Ratio of thermal voltage to typical ohmic drop in the negative electrode film (Ref#3 eq A.14) ##

    % Store scaling factors
    sf.gamma_p = cp_max/cn_max;            % Ratio of maximum lithium concentrations in electrode to maximum concentration in negative electrode
    sf.CC_rp = F./(kp*ap*sqrt(ce0)*t0);    % Radius of active material particle
    sf.i0 = i0;
    sf.t0 = t0;
    sf.Sigma_n = R*Tref*sigma_n/(F*L*i0);   % Ratio of thermal voltage to typical ohmic drop in the negative electrode (Ref#1 eq A.9)
    sf.Sigma_e = R*Tref*sigma_e/(F*L*i0);   % Ratio of thermal voltage to typical ohmic drop in the electrolyte (Ref#1 eq A.9)
    sf.Sigma_p = R*Tref*sigma_p/(F*L*i0);   % Ratio of thermal voltage to typical ohmic drop in the positive electrode (Ref#1 eq A.9)
    sf.gamma_T = R*cn_max/Cp;               % Ratio of temperautre variation to reference temperature
    sf.beta_sr = an*Lf0;
    sf.Sigma_film_n = Sigma_film_n;
    sf.lambda = lambda;
    sf. CC_rsei = CC_rsei;
    sf.CC_sei = CC_sei;
    sf.CC_rLi = CC_rLi;

    % Store varying parameters
    sf.kp = kp;
    sf.sigma_n = sigma_n;
    sf.sigma_e = sigma_e;
    sf.sigma_p = sigma_p;
    sf.That = (T - Tref)*Cp/(R*Tref*cn_max);

elseif ismember(para.model_type,"TECMD")
    R = 8.3144;       % Universal gas constant

    % Thermal parameters
    Tref = value(simscape.Value(para.Tref,para.Tref_unit),'K');
    RoRef = value(simscape.Value(para.RoRef,para.RoRef_unit),'Ohm');
    Ea_Ro = value(simscape.Value(para.Ea_Ro,para.Ea_Ro_unit),'J/mol');


    Ro = RoRef*exp(Ea_Ro/R*(1./T-1/Tref));        % Arrhenius type equation: Ro

    % Store varying parameters
    sf.Ro = Ro;
elseif ismember(para.model_type,"TECM")
    sf = [];
end
end

function signal = getSignal(states,block_path,signal_str,time_limits)

arguments
    states
    block_path
    signal_str
    time_limits
end
block_name = replace(block_path,"/",".");
dsignal = find(states,"Name",block_name + "."+ signal_str);

% Might need furture revisons when Matlab removes timeseries format and uses
% only timetable foramt
data_timeseries = dsignal{1}.Values.getsampleusingtime(time_limits(1),time_limits(2));
signal = data_timeseries.Data;

end

function V = getVoltage(states,block_path,para,SoC,time_limits)
arguments
    states
    block_path
    para
    SoC
    time_limits
end

if para.model_type == "TECM"
    U = getSignal(states,block_path,"OCV",time_limits);
    Ro = getSignal(states,block_path,"Ro",time_limits);
    etaP = getSignal(states,block_path,"etaP",time_limits);
    try
        v = getSignal(states,block_path,"v",time_limits);
        etaO = v - U - etaP;
    catch
        i = getSignal(states,block_path,"i",time_limits);
        etaO = i.*Ro;
        v = U + etaO + etaP;
    end
    V.V = v;
    V.U = U;
    V.eta0 = etaO;
    V.etaP = etaP;

elseif para.model_type == "TECMD"

    U = getSignal(states,block_path,"OCV",time_limits);
    etaO = getSignal(states,block_path,"etaO",time_limits);
    etaP = getSignal(states,block_path,"etaP",time_limits);
    try
        v = getSignal(states,block_path,"v",time_limits);
        etaC = v - U - etaO - etaP;
    catch
        etaC = getSignal(states,block_path,"etaC",time_limits);
        v = U + etaO + etaP + etaC;
    end

    V.V = v;
    V.U = U;
    V.eta0 = etaO;
    V.etaP = etaP;
    V.etaC = etaC;
end

end
