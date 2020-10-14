function delta_f=df_over_f(cell_trace,Frames_p_acq)
lowest_values=10; %how many lowest values taken as F0
for ii=1:(size(cell_trace,2)/Frames_p_acq)
    sorted_data((1+Frames_p_acq*(ii-1)):(ii*Frames_p_acq))=sort(cell_trace((1+Frames_p_acq*(ii-1)):(ii*Frames_p_acq)));
    F0=mean(sorted_data((1+Frames_p_acq*(ii-1)):(lowest_values+Frames_p_acq*(ii-1)))); %F0 based on the ten lowest values
    delta_f((1+Frames_p_acq*(ii-1)):(ii*Frames_p_acq))=(cell_trace((1+Frames_p_acq*(ii-1)):(ii*Frames_p_acq))-F0)/F0;
 
end
   delta_f=delta_f*100; %change to percentage
end

