clear all
close all

%% Written by Marcin Szczot, PhD
%% NIH/NCCIH
%% current address marcin.szczot@liu.se
%% please cite Ghitani et al. 2016 Neuron

run('initiate_options.m');
run('Initial_neuropil.m');
ctrl=[]
while isempty(ctrl);                % query user for load and save pathways
    pause(0.1);
    D=get(Initial_neuropil,'Children');
    ctrl=D(3).UserData;
end
close(Initial_neuropil)

%% load ROI, do some initial calculations with them
ROI_list=dir(strcat(Load_path,filesep,'*.roi'))         %load ROI list
if options_np.long_flag==1
    % setup breakpoint at line 17
    %rerun line: movie_name=dir(strcat(Load_path,filesep,'*.tif'))
    %run line movie=bigread2(strcat(Load_path,filesep,movie_name.name),1);
    %with different names as in following examples:
    %IMPORTANT note for MAC!!!! some file names might duplicated with a dot (.)
    %at the beginning of the file name!!!! in that case you have to be able to read how many actual files are there 
    %and only concatenate in the correct order the real tiff files without this stupid dot.
    % sample for movie 1:   movie1=bigread2(strcat(Load_path,filesep,movie_name(1).name),1);
    % sample for movie 2    movie2=bigread2(strcat(Load_path,filesep,movie_name(2).name),1);
    % redo for however many movies you have and then concatenate them as
    % variable 'movie';
    % movie=cat(3,movie1,movie2);
    % after your done delete temporary movie variables
else
    movie_name=dir(strcat(Load_path,filesep,'*.tif'))
    mov_info=imfinfo(strcat(Load_path,filesep,movie_name.name));
    movie=bigread2(strcat(Load_path,filesep,movie_name.name),1); %in the end substitute to read all frames
end
max_width=0;
max_heigth=0;
for i=1:max(size(ROI_list))
    ROI_data{i}=ReadImageJROI(strcat(Load_path,filesep,ROI_list(i).name));
    if (ROI_data{i}.vnRectBounds(3)-ROI_data{i}.vnRectBounds(1))>max_width;
        max_width=(ROI_data{i}.vnRectBounds(3)-ROI_data{i}.vnRectBounds(1));
    end
    if (ROI_data{i}.vnRectBounds(4)-ROI_data{i}.vnRectBounds(2))>max_heigth;
        max_heigth=(ROI_data{i}.vnRectBounds(4)-ROI_data{i}.vnRectBounds(2));
    end
end
%% run and setup initial GUI values
run('GUI_neuropil.m');
State=guidata(GUI_neuropil);
State.Expansion_size_val.String=options_np.expansion_size;
State.Exclusion_size_val.String=options_np.exclusion_size;
State.edit5.String=options_np.exclusion_percentage;
State.Exclusion_percentage_val.String=options_np.exclusion_percentage;
State.text9.String=num2str(max(size(ROI_list)));
State.text9.UserData=ROI_list;
State.Back_to_default.UserData=options_np;
pad_size=round(0.6*(max([max_width max_heigth]))*size_leak);
for i=1:max(size(ROI_list))
    ROI_data{i}.vnRectBounds=ROI_data{i}.vnRectBounds+pad_size;
    ROI_data{i}.mnCoordinates=ROI_data{i}.mnCoordinates+pad_size;
end
movie=padarray(movie,[pad_size pad_size]);
State.text5.UserData=movie;
State.text6.UserData=[max_width max_heigth];
State.text7.String=num2str(1);
State.movie_slide.Min=1;
State.movie_slide.Max=size(movie,3);
State.movie_slide.SliderStep=[1/(size(movie,3)-1) 50/(size(movie,3)-1)];
State.movie_slide.Value=1;
State.pushbutton2.UserData=cell(1,3);
State.pushbutton3.UserData=cell(1,2);
State.pushbutton3.UserData{1,1}=1;
State.pushbutton3.UserData{1,2}=cell(size(ROI_list,1)+1,1);
State.pushbutton2.UserData{1,1}=1;
State.pushbutton2.UserData{1,2}(:,State.pushbutton2.UserData{1,1})=linspace(1,size(movie,3),size(movie,3))';
State.pushbutton2.UserData{1,3}=cell(size(ROI_list,1)+1,1);
State.text8.UserData=cell(1,2);
State.text8.UserData{1,1}=Load_path;
State.text8.UserData{2,1}=Save_path;
State.radiobutton1.Value=1;
State.raw_trace.XTickLabel=[];
State.raw_trace.Title.String='Raw df/f';
State.neuropil_trace.XTickLabel=[];
State.neuropil_trace.Title.String='Neuropil df/f';
State.result_trace.Title.String='Total df/f';
State.axes_local_movie.Title.String='Movie constant - color scaling';
State.axes_initial_calculation.Title.String='Max intensity projection';
h = waitbar(0,'loading cells... will take a while') 
for i=1:size(State.text9.UserData,1);
    [ROIname,small_image,ROI_small_coordinates,max_sm_image]=...
    load_n_crop(State.text9.UserData(str2num(State.text7.String)),State,size_leak,ROI_data{i});
    State.movie_slide.UserData{i}=small_image;
    State.small_im_movie.UserData{str2num(State.text7.String)}=small_image;
    State.zproj.UserData{1,str2num(State.text7.String)}=ROI_small_coordinates;
    State.zproj.UserData{2,str2num(State.text7.String)}=max_sm_image;
    State.text7.String=num2str(str2num(State.text7.String)+1);
    waitbar(i/size(State.text9.UserData,1),h,'loading...');
end
close(h);
State.text7.String=num2str(1);
colormap(State.axes_local_movie,'jet');
colormap(State.axes_initial_calculation,'jet');
colormap(State.axes_updated_calculation,'jet');
close all
cd(Load_path);
%% initialize cell (no.1)
calculate_n_draw_small(State.zproj.UserData{1,str2num(State.text7.String)},...
State.small_im_movie.UserData{str2num(State.text7.String)},State,State.movie_slide.Value,...
str2num(State.text7.String));
