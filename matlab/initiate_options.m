options_np=struct('expansion_size',10,...                   %how much to expand in search for neuropil signal
                  'exclusion_size',3,...                    %how many pixels exclude at the border
                  'exclusion_percentage',0.1,...            %how much of high level neuropil pixels exclude
                  'pause_after_graphing',1,...
                  'frames_per_acq',200,...
                  'long_flag',0)                            %set to 1 if movie is to large to load with bigread and needs to be loaded manually from splits instruction how to load from splits follows at the bottom
           % %% setting main menu GUI and decisions following setting up
           
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