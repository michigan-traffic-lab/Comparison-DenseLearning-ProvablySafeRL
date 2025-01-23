% main - Set-Based Occupancy Prediction
%
% Inputs:
%   none
% User input is defined within the code below:
%       inputFile - input file (in XML format)
%       time intervals - time stamps for the prediction and visualization
%
% Outputs:
%   perception - object that contains a map with obstacles and their
%                occupancies

%------------- BEGIN CODE --------------

function [opt_time] = spot_main(filename, dt, prediction_horizon)

    %% --- User Settings ---

    % define the input file for the traffic scenario
    inputFile = filename; %'scenarios/test.xml';
    % time interval in seconds for prediction of the occupancy
    ts_prediction = 0;
    dt_prediction = dt;
    tf_prediction = prediction_horizon;

    %% --- Set-up Perception ---

    % create perception from input (holding a map with all lanes, adjacency
    % graph and all obstacles), scenario will start at its beginning
    perception = globalPck.Perception(inputFile);
    tic;
    % Inputs:
    %   ts - starting time in s
    %   dt - time step size in s
    %   tf - ending in time in s
    % create time interval for occupancy calculation
    timeInterval_prediction = globalPck.TimeInterval(ts_prediction, dt_prediction, tf_prediction);

    %% --- Main code ---
    % --- do occupancy calculation ---
    perception.computeOccupancyGlobal(timeInterval_prediction);
    opt_time = toc;
    % --- write obstacles' occupancy ---
    % obstacles = perception.map.obstacles;
    output.writeToXML(perception.map, filename, 2);
end