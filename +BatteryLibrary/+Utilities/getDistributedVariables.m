function  dv = getDistributedVariables(simout, block_names, nvp)
% Get the distibuted variables of TECMD, TSPMe and TSPMeA
%
% W.D.Widanage 24/05/2023 (Nothing else matters)
 
arguments
    simout Simulink.SimulationOutput 
    block_names {mustBeText}
    nvp.state_name = "xout";
end
try
    simout.(char(nvp.state_name));
    state_name = nvp.state_name;
catch
    error(sprintf("The states are not saved in the current Simulink model. \nPlease go to the Siumlink ""Model Settings"" and in the ""Data Import/Export"" section enable the ""States"" check box option in ""Save to workspace"" and re-run model" ))
end

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

% Get dialog parameters of each of the desired block_names and plot
% distributed varaibles
for bb = 1:numel(block_names)

    block_name = block_names(bb);

    if  ismember(block_name,simout_block_names)
        [~,blk_idx] = ismember(block_name,simout_block_names);
        block_path = simout_block_path_list(blk_idx);
        model_para = BatteryLibrary.Utilities.getModelParameters(block_path);

        % Get the type of the simscape battery block: TECMD, TSPMe or TSMPeA
        model_type = model_para.(block_name).model_type;
        
         r = getParticleDims(model_para.(block_name));
         x = getElectrodeSeperatorDims(model_para.(block_name));


         if ismember(model_type,["TSPMe","TSPMeA"])
             c = getConcetrations(simout_states,block_path,model_para.(block_name));
             dv.(block_name) = c;
             % phi = getPotentials(simout,block_path);
             % J = getSideReactions(simout,block_path);
             % eps = getPorosity(simput,block_path);
         end

    else
        simout_block_names_tmp = split(simout_block_path_list(bb),"/");
        simulink_filename = simout_block_names_tmp(1);
        warning(sprintf("A block named %s is not present in the simulink model %s.slx",block_names(bb),simulink_filename))
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
x.n = xn*Ln_SI;                      % Negative electrode domain [0, Ln]
x.p = xp*Lp_SI;                      % Positive electrode domain [Ln+Ls, Lp]
x.x = [xn,xs(2:end-1),xp]*L_SI;      % Full electrode domain [0,L]

end

function c = getConcetrations(states,block_path,para)

arguments
    states, 
    block_path
    para
end

cn_max = value(simscape.Value(para.cn_max,para.cn_max_unit),'mol/m^3');
cp_max = value(simscape.Value(para.cp_max,para.cp_max_unit),'mol/m^3');
ce0 = value(simscape.Value(para.ce0,para.ce0_unit),'mol/m^3');

block_name = replace(block_path,"/",".");
dcn = find(states,"Name",block_name + ".cn_hat");
dcp = find(states,"Name",block_name + ".cp_hat");

dcen = find(states,"Name",block_name + ".cen_hat");
dces = find(states,"Name",block_name + ".ces_hati");
dcep = find(states,"Name",block_name + ".cep_hat");

% Might need furture revisons when Matlab removes timeseries format and uses
% only timetable foramt
c.cn = dcn{1}.Values.Data*cn_max;
c.cp = dcp{1}.Values.Data*cp_max;
c.ce = [dcen{1}.Values.Data, dces{1}.Values.Data, dcep{1}.Values.Data]*ce0;
end