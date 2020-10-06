function [ROIname,result_trace]=calculate_n_draw(filename,handles,frame_no)
size_leak=3;   %how much bigger the field of view should be relative to cell
a=cd;
%% import data from GUI objects and external files
movie=handles.text5.UserData;
ROI_data=ReadImageJROI(strcat(a,filesep,filename.name));
ROIrectangle=ROI_data.vnRectBounds;
ROIname=ROI_data.strName;
maxCells=handles.text6.UserData;
%% set cropping boundaries
image_center=[(ROIrectangle(2)+ROIrectangle(4))/2 (ROIrectangle(1)+ROIrectangle(3))/2];
%field_size=(ROIrectangle(3)-ROIrectangle(1))*size_leak/2;
crop_size=[image_center(1)-0.5*size_leak*maxCells(1) image_center(2)-0.5*size_leak*maxCells(2) maxCells(1)*size_leak maxCells(2)*size_leak];

%% crop both movies and ROI to small size
for i=1:size(movie,3)
small_image(:,:,i)=imcrop(movie(:,:,i),round(crop_size));
handles.movie_slide.UserData=small_image;
end
max_sm_image=max(small_image,[],3);
small_image(1:3,1:3,:)=max(max(max(small_image)));
small_image(4:6,1:3,:)=min(min(min(small_image)));

ROI_small_coordinates=[ROI_data.mnCoordinates(:,1)-round(crop_size(1)) ROI_data.mnCoordinates(:,2)-round(crop_size(2))]; 
ConvROI=convhull(ROI_small_coordinates(:,1),ROI_small_coordinates(:,2));
outer_donut=round(expanded_ROI(ROI_small_coordinates(ConvROI,1),ROI_small_coordinates(ConvROI,2),str2num(handles.Expansion_size_val.String)));
inner_donut=round(expanded_ROI(ROI_small_coordinates(ConvROI,1),ROI_small_coordinates(ConvROI,2),str2num(handles.Exclusion_size_val.String)));
%% prepare donut-size mask in original coordinates
Louter_mask=poly2mask(outer_donut(:,1)+round(crop_size(1)),outer_donut(:,2)+round(crop_size(2)),max(size(movie(:,1,1))),max(size(movie(1,:,1))));
Linner_mask=poly2mask(inner_donut(:,1)+round(crop_size(1)),inner_donut(:,2)+round(crop_size(2)),max(size(movie(:,1,1))),max(size(movie(1,:,1))));
Lcell_mask=poly2mask(ROI_data.mnCoordinates(:,1),ROI_data.mnCoordinates(:,2),max(size(movie(:,1,1))),max(size(movie(1,:,1))));
Ldonut_mask=Louter_mask-Linner_mask;

%test line for adjustment if needed
%Sdonut_mask=imcrop(Ldonut_mask,round(crop_size));

%% calculate fluorescence traces in a cell and full donut (using full picture)
sparse_Ldonut=sparse(Ldonut_mask);
sparse_cell=sparse(Lcell_mask);
[rowD,colD] = find(sparse_Ldonut);
[rowC,colC] = find(sparse_cell);
    for i=1:max(size(rowC))             %calculate average trace for the cell
        pix_cell_trace(i,:)=movie(rowC(i),colC(i),:);
    end
cell_trace=mean(pix_cell_trace,1);
if isempty(rowD)
    h = msgbox(['Those settings give no pixels for donut! try different-' 'no donut will be graphed-' 'neuropil if saved will be considered-' 'zero vector']);
    average_neuropil=zeros(1,max(size(movie(1,1,:))));
else
    for i=1:max(size(rowD))
        donut_trace(i,:)=movie(rowD(i),colD(i),:);
        max_tr_donut(i)=max(donut_trace(i,:));
        position(i,:)=[rowD(i) colD(i)];
    end
    
    [include]=find(max_tr_donut<=(1-str2num(handles.Exclusion_percentage_val.String))*max(max_tr_donut) &... %for max smaller than given perc. of max
        max_tr_donut>=(1+str2num(handles.Exclusion_percentage_val.String))*min(max_tr_donut));    %for max larger than given perc. of minimal max
  if isempty(include)
    h = msgbox(['Those settings give no pixels for donut! try different-' 'no donut will be graphed-' 'neuropil if saved will be considered-' 'zero vector']);
    average_neuropil=zeros(1,max(size(movie(1,1,:))));
  else
    average_neuropil=mean(donut_trace(include,:));
    Ldonut_mask_corr=zeros(size(Ldonut_mask));
    for i=1:max(size(include))
        try
        Ldonut_mask_corr(rowD(include(i)),colD(include(i)))=1;
        catch
            a=1
        end
    end
    Sdonut_mask_corr=imcrop(Ldonut_mask_corr,round(crop_size));
    %% manually!!! (there should be function for this! calculating edges for display purposes
    Ldonut_mask_corr_outl=zeros(size(Ldonut_mask));
    for i=1:max(size(include))
        if rowD(include(i))==1 || colD(include(i))==1
            Ldonut_mask_corr_outl(rowD(include(i)),colD(include(i)))=1;
        elseif Ldonut_mask_corr(rowD(include(i))-1,colD(include(i))-1)==1 &&...
                Ldonut_mask_corr(rowD(include(i))-1,colD(include(i)))==1 && ...
                Ldonut_mask_corr(rowD(include(i))-1,colD(include(i))+1)==1 &&...
                Ldonut_mask_corr(rowD(include(i)),colD(include(i))-1)==1 &&...
                Ldonut_mask_corr(rowD(include(i)),colD(include(i))+1)==1 &&...
                Ldonut_mask_corr(rowD(include(i))+1,colD(include(i))-1)==1 &&...
                Ldonut_mask_corr(rowD(include(i))+1,colD(include(i)))==1 && ...
                Ldonut_mask_corr(rowD(include(i))+1,colD(include(i))+1)==1
            Ldonut_mask_corr_outl(rowD(include(i)),colD(include(i)))=0;
        else
            Ldonut_mask_corr_outl(rowD(include(i)),colD(include(i)))=1;
        end
    end
    Sdonut_mask_corr_outl=imcrop(Ldonut_mask_corr_outl,round(crop_size));
  end
end
%% check whether images are already there if not create them if they are substitute with updated ones (same goes for patches)
if isempty(handles.axes_local_movie.UserData)
    sm_local_movie=imagesc(small_image(:,:,frame_no),'Parent',handles.axes_local_movie);
    initial_calculation=imagesc(max_sm_image,'Parent',handles.axes_initial_calculation);
    updated_calculation=imagesc(max_sm_image,'Parent',handles.axes_updated_calculation);
    handles.axes_local_movie.UserData=sm_local_movie;
    handles.axes_initial_calculation.UserData=initial_calculation;
    handles.axes_updated_calculation.UserData= updated_calculation;
% all patch objects get +1 to synchronize with image (because images start
% at one but graphs start at 0 (!!!);
    handles.ROI_data_holder.UserData(1)=patch(ROI_small_coordinates(:,1)+1,ROI_small_coordinates(:,2)+1,[1 0 0],...
        'FaceAlpha',0,'Parent',handles.axes_local_movie);
    handles.ROI_data_holder.UserData(2)=patch(ROI_small_coordinates(:,1)+1,ROI_small_coordinates(:,2)+1,[1 0 0],...
        'FaceAlpha',0,'Parent',handles.axes_initial_calculation);
    handles.ROI_data_holder.UserData(3)=patch(ROI_small_coordinates(:,1)+1,ROI_small_coordinates(:,2)+1,[1 0 0],...
        'FaceAlpha',0,'Parent',handles.axes_updated_calculation);    
else
   sm_local_movie= handles.axes_local_movie.UserData;
   initial_calculation=handles.axes_initial_calculation.UserData;
   updated_calculation=handles.axes_updated_calculation.UserData;
   sm_local_movie.CData=small_image(:,:,round(frame_no));
   if ~isempty(include)
   max_sm_image=max_sm_image+80*uint8(Sdonut_mask_corr_outl); %test line change at the first possible moment;
   end
   initial_calculation.CData=max_sm_image;
   updated_calculation.CData=max_sm_image;
   delete(handles.ROI_data_holder.UserData(1));
   handles.ROI_data_holder.UserData(1)=patch(ROI_small_coordinates(:,1)+1,ROI_small_coordinates(:,2)+1,[1 0 0],...
        'FaceAlpha',0,'Parent',handles.axes_local_movie);
   delete(handles.ROI_data_holder.UserData(2));
   handles.ROI_data_holder.UserData(2)=patch(ROI_small_coordinates(:,1)+1,ROI_small_coordinates(:,2)+1,[1 0 0],...
        'FaceAlpha',0,'Parent',handles.axes_initial_calculation);
   delete(handles.ROI_data_holder.UserData(3));
   handles.ROI_data_holder.UserData(3)=patch(ROI_small_coordinates(:,1)+1,ROI_small_coordinates(:,2)+1,[1 0 0],...
        'FaceAlpha',0,'Parent',handles.axes_updated_calculation);
   delete(handles.ROI_data_holder.UserData(4));
   delete(handles.ROI_data_holder.UserData(5));
   delete(handles.ROI_data_holder.UserData(6));
   delete(handles.ROI_data_holder.UserData(7));
   delete(handles.ROI_data_holder.UserData(8));
   delete(handles.ROI_data_holder.UserData(9));
end
handles.ROI_data_holder.UserData(4)=patch(outer_donut(:,1)+1,outer_donut(:,2)+1,[1 0 0],...
    'FaceAlpha',0,'Parent',handles.axes_local_movie,'EdgeColor',[1 0 0]);
handles.ROI_data_holder.UserData(5)=patch(inner_donut(:,1)+1,inner_donut(:,2)+1,[1 0 0],...
    'FaceAlpha',0,'Parent',handles.axes_local_movie,'EdgeColor',[1 0 0]);
handles.ROI_data_holder.UserData(6)=patch(outer_donut(:,1)+1,outer_donut(:,2)+1,[1 0 0],...
    'FaceAlpha',0,'Parent',handles.axes_initial_calculation,'EdgeColor',[1 0 0]);
handles.ROI_data_holder.UserData(7)=patch(inner_donut(:,1)+1,inner_donut(:,2)+1,[1 0 0],...
    'FaceAlpha',0,'Parent',handles.axes_initial_calculation,'EdgeColor',[1 0 0]);
%test line for assuring that all adjustments ane rescalings are correct
%handles.ROI_data_holder.UserData(3)=imagesc(Sdonut_mask,'Parent',handles.axes_updated_calculation,'AlphaData',0.5);
handles.ROI_data_holder.UserData(8)=patch(outer_donut(:,1)+1,outer_donut(:,2)+1,[1 0 0],...
    'FaceAlpha',0,'Parent',handles.axes_updated_calculation,'EdgeColor',[1 0 0]);
handles.ROI_data_holder.UserData(9)=patch(inner_donut(:,1)+1,inner_donut(:,2)+1,[1 0 0],...
    'FaceAlpha',0,'Parent',handles.axes_updated_calculation,'EdgeColor',[1 0 0]);
%% calculate resultant traces and draw them
if ~isempty(handles.raw_trace.UserData)
    delete(handles.raw_trace.UserData(1));
    delete(handles.raw_trace.UserData(2));
    delete(handles.raw_trace.UserData(3));
end
x=linspace(1,size(movie,3),size(movie,3));
result_trace=cell_trace-average_neuropil;
min_f=min([cell_trace average_neuropil]);
handles.raw_trace.UserData(1)=plot(handles.raw_trace,x,cell_trace,'Color',[1 0 0]);
handles.raw_trace.UserData(2)=plot(handles.neuropil_trace,x,average_neuropil,'Color',[0 1 0]);
handles.raw_trace.UserData(3)=plot(handles.result_trace,x,result_trace,'Color',[0 0 1]);
set(handles.raw_trace,'YLim',[0.8*min_f 1.2*max(cell_trace)]);
set(handles.neuropil_trace,'YLim',[0.8*min_f 1.2*max(cell_trace)]);
set(handles.result_trace,'YLim',[0.8*min(result_trace) 1.2*max(result_trace)]);
%% try to save chosen cells we will save 1) figure with ROI 
end
