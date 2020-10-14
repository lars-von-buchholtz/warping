function [coordinates]=expanded_ROI(X,Y,size)
Temp1=expandPolygon([X Y],size);
TempX=Temp1{1}(:,1);
TempY=Temp1{1}(:,2);
TempfinX=TempX(isfinite(TempX));
TempfinY=TempY(isfinite(TempY));
coordinates=[TempfinX TempfinY];
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
end

