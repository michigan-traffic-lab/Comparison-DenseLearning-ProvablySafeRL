% main - Set-Based Occupancy Prediction
%
% Syntax:
%   main()
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
%
% Other m-files required:
% Subfunctions:
% MAT-files required:
%
% See also: M. Koschi and M. Althoff, 2017, SPOT: A Tool for Set-Based
% Prediction of Traffic Participants; and
% M. Althoff and S. Magdici, 2016, Set-Based Prediction of Traffic
% Participants on Arbitrary Road Networks

% Author:       Markus Koschi
% Written:      06-September-2016
% Last update:  08-June-2017
%
% Last revision:---

%------------- BEGIN CODE --------------


%% --- Preliminaries ---

clc;
clear;
close all;


%% --- User Settings ---

% define the input file for the traffic scenario
% scenarios of CommonRoad:
%inputFile = 'scenarios/NGSIM_US101_0.xml';
%inputFile = 'scenarios/NGSIM_US101_4.xml';
%inputFile = 'scenarios/NGSIM_US101_5.xml';
%inputFile = 'scenarios/GER_A9_2a.xml';
%inputFile = 'scenarios/GER_B471_1.xml';
%inputFile = 'scenarios/GER_Ffb_1.xml';
%inputFile = 'scenarios/GER_Gar_1.xml';
%inputFile = 'scenarios/GER_Muc_3a.xml';
inputFile = 'scenarios/test3.xml';
% other examples:
%inputFile = 'scenarios/Intersection_Leopold_Hohenzollern_v3.osm';
%inputFile = 'scenarios/Fuerstenfeldbruck_T_junction.osm';
%inputFile = 'scenarios/2lanes_same_merging_1vehicle_egoVehicle.osm';
%inputFile = 'scenarios/2lanes_same_merging_2vehicles.osm';


% time interval of the scenario (step along trajectory of obstacles)
% (if ts == [], scenario will start at its beginning;
% if tf == [], scenario will run only for one time step)
% ToDO: tf to run the scenario until its end
ts_scenario = [];
% (dt_scenario is defined by given trajectory)
tf_scenario = [];

% time interval in seconds for prediction of the occupancy
ts_prediction = 0;
dt_prediction = 0.1;
tf_prediction = 0.8;

% time interval in seconds for visualization
ts_plot = ts_prediction;
dt_plot = dt_prediction;
tf_plot = tf_prediction;

% define whether trajectory shall be verifyed (i.e. checked for collision)
verify_trajectory = false;

% define output
writeOutput = true;


%% --- Set-up Perception ---

% create perception from input (holding a map with all lanes, adjacency
% graph and all obstacles)
perception = globalPck.Perception(inputFile);

% Inputs:
%   ts - starting time in s
%   dt - time step size in s
%   tf - ending in time in s
% create time interval for occupancy calculation
timeInterval_prediction = globalPck.TimeInterval(ts_prediction, dt_prediction, tf_prediction);

perception.computeOccupancyGlobal(timeInterval_prediction);

disp(perception.map.obstacles(3).occupancy)

%% --- Output ---
if 1
    % save the map in an XML file (polygons are not triangulated)
    % (exportType: 1 = trajectory, 2 = occupancy set, 3 = probabilityDistribution)
    output.writeToXML(perception.map, './testOutput.xml', 2);
end

%------------- END CODE --------------