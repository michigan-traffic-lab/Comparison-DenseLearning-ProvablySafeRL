function [occupancyNode] = exportOccupancy (doc,occupancyList,iS,iF)
% exportOccupancy - returns occupancy information as a DOM tree node
%
% Syntax:
%   exportoccupancy(doc,occupancyList,iS,iF)
%
% Outputs:
%   occupancyNode - DOM tree node containing all information about the
%   occupancy
% 
%
% Other m-files required: exportShape.m


% Author:       Lukas Willinger
% Written:      12 April 2017
% Last update:
%
% Last revision:---
%
%------------- BEGIN CODE --------------

    % Create occupancySet node
    occupancyNode = doc.createElement('occupancySet');
    shapeList = geometry.Shape.empty();

    for i = iS:iF
        % save timeInterval start and end values
        [tS,~,tF] = occupancyList(1,i).timeInterval.getTimeInterval();

        shape_num = 0;
        % Traverse column and append polygons
        for j = 1:size(occupancyList,1)
            if ~isempty(occupancyList(j,i).vertices)
                shape_num = shape_num + 1;
                shapeList(end+1) = geometry.Polygon(occupancyList(j,i).vertices);
            end
        end
        shape_num = max(shape_num, 1);
        
        % create shape node
        occupancy = doc.createElement('occupancy');
        occupancyNode.appendChild(occupancy);
        occupancy.appendChild(output.exportShape(doc,shapeList(end-shape_num+1:end)));
        
        % Create timeNode
        timeNode = doc.createElement('time');
        occupancy.appendChild(timeNode);
        
        % Create time interval
        intervalStart = doc.createElement('intervalStart');
        %intervalStart.appendChild(doc.createTextNode(num2str(tS)));
        intervalStart.appendChild(doc.createTextNode(num2str(i-1)));
        timeNode.appendChild(intervalStart);
        intervalEnd = doc.createElement('intervalEnd');
        %intervalEnd.appendChild(doc.createTextNode(num2str(tF)));
        intervalEnd.appendChild(doc.createTextNode(num2str(i)));
        timeNode.appendChild(intervalEnd);
        
    end
end 

%------------- END CODE --------------