% To run this code in MATLAB the 2021 release of the CORA toolbox available
% at https://cora.in.tum.de is required

% System Dynamics ---------------------------------------------------------

% dynamic function
f = @(x,u,w) [cos(x(4))*x(3); ...
              sin(x(4))*x(3);
              u(1) + w(1); ...
              u(2) + w(2)];
          
% names of the system states
param.states = {'xPosition','yPosition','velocity','orientation'};
        
          
% Occupancy Set -----------------------------------------------------------

% function mapping system states to the space occupied by the vehicle
o = @(x,d) [x(1) + cos(x(4))*d(1) - sin(x(4))*d(2); 
            x(2) + cos(x(4))*d(2) + sin(x(4))*d(1)];
        
% dimensions of the car
l = 5;                       % length of the car
w = 1.8;                        % width of the car
    
param.D = interval([-l/2;-w/2],[l/2;w/2]);

        
% Parameter ---------------------------------------------------------------

% set of control commands
param.U = interval([-4; -0.2],[2; 0.2]);

% set of measurement errors
param.V = interval(zeros(4,1));

% set of disturbances
param.W = interval([-0.5;-0.02],[0.5;0.02]);


% Template Sets -----------------------------------------------------------

% indices of the invariant states
param.invStates = [1, 2, 4];

% initial sets for the non-invariant states
initSet = {};
bounds = 0:1:60;

for i = 1:length(bounds)-1
    initSet{end+1} = interval(bounds(i),bounds(i+1));
end

param.initSet = initSet;
    

% Motion Planning Settings ------------------------------------------------

% planning horizon
settings.tFinal = 0.8;

% number of time steps
settings.N = 2;


% Reachability Settings ---------------------------------------------------

options.timeStep = 0.01;
options.zonotopeOrder = 20;
options.taylorTerms = 10;
options.alg = 'poly';
options.tensorOrder = 3;
options.errorOrder = 5;
options.intermediateOrder = 20;


% Reachability Analysis ---------------------------------------------------

precomputeReachableSetsFeedback('commonRoad3',f,o,param,settings,options);