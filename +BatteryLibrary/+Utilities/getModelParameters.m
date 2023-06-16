function [modelStr_para] = getModelParameters(block_paths)
% Get the dialogue parameters of BatterySimscape_Library models
%  Input:
%   A string array or cell array of paths to the blocks from which the paramters need to be extracted
%   Can also use matlabs find_system function to generate a cell array of
%   block paths, e.g. >>find_system(gcs,'Type','Block'), where gcs is an
%   inbuilt matlab keyword for get current system, and then locate the
%   corresponding battery library blocks of interest from the resulting
%   list
%
%. W.D. Widanage 29/03/2023 (Is the end nigh?)

arguments
    block_paths {mustBeText} 
end

block_paths_string = string(block_paths);

for bb = 1:numel(block_paths)
    block_path = block_paths_string(bb);
    block_path_split = split( block_path,"/");
    model_name = block_path_split(end);
    if getSimulinkBlockHandle(block_path) > 0
        model_type = getModelType(block_path);
        modelStr_para.(model_name) = getBlockValuesUnits(block_path);
        modelStr_para.(model_name).model_type = model_type;
        if ismember(model_type, ["TSPMe","TSPMeA"])
            modelStr_para.(model_name).s6 = [-1	-0.809016994374948	-0.309016994374947	0.309016994374947	0.809016994374948	1];                                                         % Six collocation points used in TSPMe on standard interval [-1,1] 
            modelStr_para.(model_name).s10 = [-1	-0.939692620785908	-0.766044443118978	-0.5	-0.173648177666930	0.173648177666930	0.5	0.766044443118978	0.939692620785908	1]; % Ten collocation points used in TSPMe on standard interval [-1,1]
            modelStr_para.(model_name).nE = 10;
            modelStr_para.(model_name).nS = 6;
        end
        if ismember(model_type, "TECMD")
            modelStr_para.(model_name).xn = [0, 0.0954915028125263, 0.345491502812526, 0.654508497187474, 0.904508497187474, 1]; % Six collocation points used in TECM on interval [0,1]                                                                                                                                  
        end
    else
        warning(sprintf("No block labelled '%s' in '%s.slx'",model_names(bb),modelStr))
    end
end


end

function model_type = getModelType(block_path)

% Identify if model is TECM, TECMD, TSPMe or TSMPMeA
try
    get_param(block_path,'DialogParameters').RoLUT;
    model_type = "TECM";
end

try
    get_param(block_path,'DialogParameters').TauRef;
    model_type = "TECMD";
end

try
    get_param(block_path,'DialogParameters').DnRef;
    model_type = "TSPMe";
end

try
    get_param(block_path,'DialogParameters').DseiRef;
    model_type = "TSPMeA";
end

end

function para = getBlockValuesUnits(block_path)
block_dialog_parameters = get_param(block_path,"DialogParameters");

field_names = fieldnames(block_dialog_parameters);
for ff = 1:numel(field_names)
    field_name = field_names{ff};

    if contains(field_name ,"conf"), continue; end
    para.(field_name) =  get_param(block_path,"value@"+field_name);

end



end