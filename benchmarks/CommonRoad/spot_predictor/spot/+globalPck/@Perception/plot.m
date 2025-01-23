function plot(varargin)
% plot - plots a Perception object
%
% Syntax:
%   plot(obj, timeInterval)
%
% Inputs:
%   obj - Perception object
%   timeInterval - TimeInterval object
%
% Outputs:
%   none
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none

% Author:       Markus Koschi
% Written:      15-November-2016
% Last update:
%
% Last revision:---

%------------- BEGIN CODE --------------

if nargin == 1
    % plot the perception for the current time
    obj = varargin{1};
    timeInterval = globalPck.TimeInterval.empty();
elseif nargin == 2
    % plot the perception for the time interval
    obj = varargin{1};
    timeInterval = varargin{2};
end

% plot the percecption property map
obj.map.plot(timeInterval);

% show grid in plot
if globalPck.PlotProperties.SHOW_GRID
    grid on
else
    grid off
end

if ~globalPck.PlotProperties.SHOW_AXIS
    axis off
end

% edit axis labels
%ax = gca;
% ax.XTick = 0:10:100;
% xticklabels = 0:length(axis.XTick);
% ax.XTickLabel = {xticklabels};
%ax.YTick = 0:4.5:4.5; %PNR
%ax.YTickLabel
%ax.XTick = -50:50:400;

% edit axis
%axis fill
%axis normal
axis equal

%hold off
end

%------------- END CODE --------------