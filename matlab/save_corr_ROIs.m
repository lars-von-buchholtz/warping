function save_corr_ROIs(corr_list,corr_dir,Load_path)
%parent=strcat(Save_path,filesep);
csize=1
for i=1:size(corr_list,1)
    if ~isempty(corr_list{i})
        file=strcat(Load_path,filesep,corr_list{i},'.roi');
        list{csize}=file;
        csize=csize+1;
    end
end
zip(strcat(corr_dir,filesep,'analyzedROI.zip'),list);
end