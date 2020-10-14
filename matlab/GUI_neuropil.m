function varargout = GUI_neuropil(varargin)
% GUI_NEUROPIL MATLAB code for GUI_neuropil.fig
%      GUI_NEUROPIL, by itself, creates a new GUI_NEUROPIL or raises the existing
%      singleton*.
%
%      H = GUI_NEUROPIL returns the handle to a new GUI_NEUROPIL or the handle to
%      the existing singleton*.
%
%      GUI_NEUROPIL('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GUI_NEUROPIL.M with the given input arguments.
%
%      GUI_NEUROPIL('Property','Value',...) creates a new GUI_NEUROPIL or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GUI_neuropil_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GUI_neuropil_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GUI_neuropil

% Last Modified by GUIDE v2.5 12-Jun-2016 17:45:41

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GUI_neuropil_OpeningFcn, ...
                   'gui_OutputFcn',  @GUI_neuropil_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before GUI_neuropil is made visible.
function GUI_neuropil_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GUI_neuropil (see VARARGIN)

% Choose default command line output for GUI_neuropil
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes GUI_neuropil wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = GUI_neuropil_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function Expansion_size_val_Callback(hObject, eventdata, handles)
calculate_n_draw_small(handles.zproj.UserData{1,str2num(handles.text7.String)},...
    handles.small_im_movie.UserData{str2num(handles.text7.String)},handles,handles.movie_slide.Value,...
    str2num(handles.text7.String));

function Expansion_size_val_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Exclusion_size_val_Callback(hObject, eventdata, handles)
    calculate_n_draw_small(handles.zproj.UserData{1,str2num(handles.text7.String)},...
    handles.small_im_movie.UserData{str2num(handles.text7.String)},handles,handles.movie_slide.Value,...
    str2num(handles.text7.String));

function Exclusion_size_val_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Exclusion_percentage_val_Callback(hObject, eventdata, handles)
    if handles.radiobutton1.Value==1;
    handles.edit5.String=handles.Exclusion_percentage_val.String;
    end
    calculate_n_draw_small(handles.zproj.UserData{1,str2num(handles.text7.String)},...
    handles.small_im_movie.UserData{str2num(handles.text7.String)},handles,handles.movie_slide.Value,...
    str2num(handles.text7.String));

function Exclusion_percentage_val_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Back_to_default_Callback(hObject, eventdata, handles)


function movie_slide_Callback(hObject, eventdata, handles)
sm_local_movie= handles.axes_local_movie.UserData;
small_image=handles.movie_slide.UserData{str2num(handles.text7.String)}...
    (:,:,round(handles.movie_slide.Value));
small_image(1:3,1:3,:)=max(max(max(handles.movie_slide.UserData{str2num(handles.text7.String)})));
small_image(4:6,1:3,:)=min(min(min(handles.movie_slide.UserData{str2num(handles.text7.String)})));
sm_local_movie.CData=small_image;
if size(handles.raw_trace.Children,1)==2
    delete(handles.raw_trace.Children(1));
    delete(handles.neuropil_trace.Children(1));
    delete(handles.result_trace.Children(1));
end
axes(handles.raw_trace);
line([round(handles.movie_slide.Value) round(handles.movie_slide.Value)],[handles.raw_trace.YLim(2) 0]);
axes(handles.neuropil_trace);
line([round(handles.movie_slide.Value) round(handles.movie_slide.Value)],[handles.neuropil_trace.YLim(2) 0]);
axes(handles.result_trace);
line([round(handles.movie_slide.Value) round(handles.movie_slide.Value)],[handles.result_trace.YLim(2) handles.result_trace.YLim(1)]);


function movie_slide_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
hObject.UserData{1,1}=hObject.UserData{1,1}+1;
a=cd;
filename=handles.text9.UserData(str2num(handles.text7.String));
ROI_name=filename.name(1:end-4)
[neurop_corr_trace]=calculate_n_draw_small(handles.zproj.UserData{1,str2num(handles.text7.String)},...
    handles.small_im_movie.UserData{str2num(handles.text7.String)},handles,handles.movie_slide.Value,...
    str2num(handles.text7.String));
temp_ref=handles.zproj.UserData{1,str2num(handles.text7.String)};
[~,area]=convhull(temp_ref(:,1),temp_ref(:,2));
hObject.UserData{1,2}(:,hObject.UserData{1,1})=neurop_corr_trace';
hObject.UserData{1,4}(1,hObject.UserData{1,1})=area;
saveas(gcf,strcat(handles.text8.UserData{2,1},filesep,ROI_name),'tif');
%if max(size(ROI_name))>=9
hObject.UserData{1,3}{hObject.UserData{1,1}}=ROI_name;   
if str2num(handles.text7.String)<str2num(handles.text9.String);
    handles.text7.String=num2str(str2num(handles.text7.String)+1);
    calculate_n_draw_small(handles.zproj.UserData{1,str2num(handles.text7.String)},...
    handles.small_im_movie.UserData{str2num(handles.text7.String)},handles,handles.movie_slide.Value,...
    str2num(handles.text7.String));
else
    h = msgbox('You are done, wait for GUI to close');
    Save_path=handles.text8.UserData{2,1};
    Load_path=handles.text8.UserData{1,1};
    traces=handles.pushbutton2.UserData{1,2};
    corr_ROI=handles.pushbutton2.UserData{1,3};
    disc_ROI=handles.pushbutton3.UserData{1,2};
    %short code to sort traces by size / area
    [~,ind]=sort(hObject.UserData{1,4}(2:end));
    ind=ind+1; %because 1st row contains numbering only
    ind=[1 ind];
    traces=traces(:,ind);
    corr_ROI=corr_ROI(ind);
    csvwrite(strcat(Save_path,filesep,'neuro_corr_traces'),traces);
    save(strcat(Save_path,filesep,'corrected_ROI_list'),'corr_ROI');
    save(strcat(Save_path,filesep,'discarded_ROI_list'),'disc_ROI');
    %with labels
    labelled=num2cell(traces);
    headers=['area in pixels' num2cell(sort(hObject.UserData{1,4}(2:end))); corr_ROI(1:max(find(~cellfun(@isempty,corr_ROI))))'];
    labelled_data=[headers; labelled];
    cell2csv(strcat(Save_path,filesep,'neuro_corr_traces_labelled.csv'),labelled_data);
    save_corr_ROIs(corr_ROI,Save_path,Load_path);
    pause(5)
    close(GUI_neuropil)
end

function pushbutton3_Callback(hObject, eventdata, handles)
a=cd;
hObject.UserData{1,1}=hObject.UserData{1,1}+1;
filename=handles.text9.UserData(str2num(handles.text7.String));
ROI_name=filename.name(1:end-4);
hObject.UserData{1,2}{hObject.UserData{1,1}}=ROI_name;  
if str2num(handles.text7.String)<str2num(handles.text9.String);
    handles.text7.String=num2str(str2num(handles.text7.String)+1);
    calculate_n_draw_small(handles.zproj.UserData{1,str2num(handles.text7.String)},...
    handles.small_im_movie.UserData{str2num(handles.text7.String)},handles,handles.movie_slide.Value,...
    str2num(handles.text7.String));
else
    h = msgbox('You are done, wait for GUI to close');
    Save_path=handles.text8.UserData{2,1};
    Load_path=handles.text8.UserData{1,1};
    traces=handles.pushbutton2.UserData{1,2};
    csvwrite(strcat(Save_path,filesep,'neuro_corr_traces'),traces);
    corr_ROI=handles.pushbutton2.UserData{1,3};
    disc_ROI=handles.pushbutton3.UserData{1,2};
    save(strcat(Save_path,filesep,'corrected_ROI_list'),'corr_ROI');
    save(strcat(Save_path,filesep,'discarded_ROI_list'),'disc_ROI');
    %with labels
    labelled=num2cell(traces);
    headers=corr_ROI(1:max(find(~cellfun(@isempty,corr_ROI))))';
    labelled_data=[headers; labelled];
    cell2csv(strcat(Save_path,filesep,'neuro_corr_traces_labelled.csv'),labelled_data);
    save_corr_ROIs(corr_ROI,Save_path,Load_path);
    pause(5)
    close(GUI_neuropil)
end

function text9_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to text9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in help.
function help_Callback(hObject, eventdata, handles)
s=['This version does have -' ... 
'correct edge effect correction -'... 
'remember though'...
'settings must be set so -' ...
'no edge is included in the donut -']
h = msgbox(s,'HELP & Version Restrictions','help');
% hObject    handle to help (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in change_defaults.
function change_defaults_Callback(hObject, eventdata, handles)
% hObject    handle to change_defaults (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function edit5_Callback(hObject, eventdata, handles)
    if handles.radiobutton1.Value==1
    handles.Exclusion_percentage_val.String=handles.edit5.String;
    end
    calculate_n_draw_small(handles.zproj.UserData{1,str2num(handles.text7.String)},...
    handles.small_im_movie.UserData{str2num(handles.text7.String)},handles,handles.movie_slide.Value,...
    str2num(handles.text7.String));


% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in radiobutton1.
function radiobutton1_Callback(hObject, eventdata, handles)
%hObject.Value=~hObject.Value;
% hObject    handle to radiobutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of radiobutton1
