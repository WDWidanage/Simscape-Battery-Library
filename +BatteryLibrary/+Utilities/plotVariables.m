function plotVariables(simout,block_names,nvp)
% Plot the time and distributed variables of the models found in the
% sismcape BatteryLibrary, once a simulation run is complete.
%
% Mandtory inputs
%   simout: Simulink.SimulationOutput type. This is the output that
%           Simulink generates in the workspage when you run a Simulink
%           model or when you call the "sim" command to simulate a
%           Simulink model.
%   block_names: Specify the names of the BatteryLibrary components that
%               were used in the simulink model as a vector of strings.
%
% Optional inouts (name-value pairs)
%   plot: "summary" (default), or a list of variables to plot
%   state_name: This is the name of the states that in the simout
%                 simulink object "xout" (default)
%   number_of_snapshots: A numeric value to indicate the number of number
%                        of snapshots to plot of the distributed
%                        variables. 10 (default)
%   time_limits: A two element numeric vector indicating a time interval
%                [s] to plot the results, useful when simulation time is
%                very large.
%
% W.D.Widanage, WMG, University of Warwick, Coventry, UK. 30/05/2023 (Interrogation)

arguments
    simout Simulink.SimulationOutput
    block_names {mustBeText}
    nvp.plot {mustBeText} = "summary"
    nvp.state_name = "xout";
    nvp.number_of_snapshots = 10;
    nvp.time_limits = [];
end

import BatteryLibrary.Utilities.*

%%%%% Only edit this section if a new signal is added %%%%%%%%%%%%%%%%%%%%%%
variables_dist = ["cn","ce","cp","Phin","Phie","Phip","eps_n","Jsei","JLi","z_dist"];
variables_time = ["V","I","T","z","z_bar","U","epsn_bar","Lfn_bar","SoH","SoH_sei","SoH_Li"];
xlabels_dist = ["rn [m]","x [m]","rp [m]","xn [m]","x [m]", "xp [m]", "xn [m]", "xn [m]", "xn [m]", "x [-]"];
ylabels_dist = ["C_n [mol/m^3]","C_e [mol/m^3]","C_p [mol/m^3]","Phi_n [V]","Phi_e [V]","Phi_p [V]","eps_n [-]","Jsei [A/m^3]","JLi [A/m^3]","SoC distributed [-]"];
ylabels_time = ["Terminal voltage [V]","Current [A]","Temperature [degC]","SoC [-]","SoC bar [-]","OCV [V]","epsn_bar [-]","Film thickness [um]","SoH [-]","SoH_sei [-]","SoH_Li [-]"];
variables_to_fields_time = ["V","I","T","SoC","SoC","V","SR","SR","SR","SR","SR"];
variables_to_fields_dist = ["c","c","c","Phi","Phi","Phi","SR","SR","SR","SoC"];
variables_to_domain_dist = ["r.n","x.x","r.p","x.n","x.x","x.p","x.n","x.n","x.n","x.x"];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
variables = [variables_dist,variables_time];
xlabels_time = repmat("time [s]",size(variables_time));
xlabels = [xlabels_dist, xlabels_time];
ylabels = [ylabels_dist, ylabels_time];
variables_to_domain_time = repmat("time",size(variables_time));
variables_to_fields = [variables_to_fields_dist, variables_to_fields_time];
variables_to_domain = [variables_to_domain_dist, variables_to_domain_time];
xlabel_dict = dictionary(variables,xlabels);
ylabel_dict = dictionary(variables,ylabels);
field_dict = dictionary(variables,variables_to_fields);
domain_dict = dictionary(variables,variables_to_domain);

TSPMeA_variables_summary = ["cn","ce","cp","Phin","Phie","Phip","eps_n","Jsei","JLi","V","I"];
TSPMe_variables_summary = ["cn","ce","cp","Phin","Phie","Phip","V","I"];
TECMD_variables_summary = ["V","I","T","z_dist"];
TECM_variables_summary = ["V","I","T","z"];


try
    simout.(char(nvp.state_name));
    state_name = nvp.state_name;
catch
    error(sprintf("The states are not saved in the current Simulink model. \nPlease go to the Siumlink ""Model Settings"" and in the ""Data Import/Export"" section enable the ""States"" check box option in ""Save to workspace"" and re-run model" ))
end

number_of_figures = numel(block_names);


for ff = 1:number_of_figures
    block_name = block_names(ff);
    number_of_experiments = numel(simout);
    t_start = 0;                            % Initialise starting time value
    for ee = 1:number_of_experiments
        dv = getVariables(simout(ee),block_name,"state_name",state_name,"time_limits",nvp.time_limits);

        if dv.(block_name).block_present == 1
            model_type = dv.(block_name).model_type;

            if nvp.plot == "summary"
                if model_type == "TECM",  plot_varaibles = TECM_variables_summary; end
                if model_type == "TECMD",  plot_varaibles = TECMD_variables_summary; end
                if model_type == "TSPMe", plot_varaibles = TSPMe_variables_summary; end
                if model_type == "TSPMeA", plot_varaibles = TSPMeA_variables_summary; end
            else
                plot_varaibles = nvp.plot;
            end
            set_of_time_plots = intersect(plot_varaibles,variables_time);
            set_of_dist_plots = intersect(plot_varaibles,variables_dist);

            number_of_time_plots = numel(set_of_time_plots);
            number_of_dist_plots = numel(set_of_dist_plots);

            time_array = dv.(block_name).time;
            if number_of_experiments > 1
                time_array = time_array + t_start;
                t_start = time_array(end);
            end
            number_of_timepoints = length(time_array);
            number_of_snapshots = nvp.number_of_snapshots;
            if (number_of_snapshots > number_of_timepoints) && (number_of_dist_plots)
                number_of_snapshots = number_of_timepoints;
                warning("Number of snapshots (%d) is larger than number of time points (%d), setting number_of_snapshots to %d",number_of_snapshots,number_of_timepoints,number_of_snapshots)
            end
            time_indices = floor(linspace(1,number_of_timepoints,number_of_snapshots));


            % Get the min and max values of the distributed variables
            for pp = 1:number_of_dist_plots
                variable_str = set_of_dist_plots(pp);
                ordinate  = dv.(block_name).(field_dict(variable_str)).(variable_str);
                min_max{pp,1} = {[min(ordinate,[],"all"),max(ordinate,[],"all")]};
            end

            if ee == 1
                figure(ff)
                plot_handle = tiledlayout("flow");
                title(plot_handle,block_name)
            end

            % Get time data sets for each of the variables
            tile_time_cntr = 1;
            for pp = 1:number_of_time_plots
                variable_str = set_of_time_plots(pp);
                ordinate  = dv.(block_name).(field_dict(variable_str)).(variable_str);
                % abscissa = dv.(block_name).(domain_dict(variable_str));
                xlabel_str = xlabel_dict(variable_str);
                ylabel_str = ylabel_dict(variable_str);

                time_figure_handle(pp) = nexttile(tile_time_cntr); hold on;
                plot(time_array,ordinate,'. -')
                xlabel(xlabel_str);
                ylabel(ylabel_str);

                tile_time_cntr = tile_time_cntr + 1;
            end

            % Get distributed data sets for each of the variables
            if number_of_dist_plots
                for tt = time_indices
                    % Plot vertical time line on the time plots
                    for vv = 1:number_of_time_plots
                        y_limits = time_figure_handle(vv).YLim;
                        time_points = [time_array(tt),time_array(tt)];
                        nexttile(vv); hold on;
                        plot(time_points,y_limits,'--',"LineWidth",1)
                    end

                    tile_dist_cntr = tile_time_cntr + 1;
                    for pp = 1:number_of_dist_plots
                        variable_str = set_of_dist_plots(pp);
                        ordinate  = dv.(block_name).(field_dict(variable_str)).(variable_str)(tt,:);
                        domain = split(domain_dict(variable_str),'.');
                        abscissa = dv.(block_name).(domain(1)).(domain(2));

                        xlabel_str = xlabel_dict(variable_str);
                        ylabel_str = ylabel_dict(variable_str);

                        nexttile(tile_dist_cntr)
                        plot(abscissa,ordinate,'. -')
                        ylim(min_max{pp}{1});
                        xlabel(xlabel_str);
                        ylabel(ylabel_str);

                        tile_dist_cntr = tile_dist_cntr + 1;
                    end
                    pause(0.05)
                end
            end
        end

    end
end
end

