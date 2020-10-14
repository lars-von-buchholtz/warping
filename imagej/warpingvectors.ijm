// install this macro for use
// it requires the TurboReg plugin (http://bigwww.epfl.ch/thevenaz/turboreg/)
// run the align_patch macro for a crude initial alignment of the stacks
// open the red/green overlay to draw the warping vectors 
// before beginning, make sure the ROI manager and Results table are empty
// activate the drawing tool by pressing the 'Record' icon (filled circle) in the toolbar
// click and drag (typically from red to green) to draw vectors
// pressing 'd' will delete the last vector
// pressing 's' will save the data (select .tsv ending to save as tab separated values)
// pressing 'l' will load saved data from a .tsv file back into the Results/ROIs


macro "Record Vectors Tool - C059o11ee" {
	leftButton=16;
	rightButton=4;
	shift=1;
	ctrl=2; 
	alt=8;
	x2=-1; y2=-1; z2=-1; flags2=-1;
	getCursorLoc(x, y, z, flags);
	//print("start");
	starting = true;
	while (flags&leftButton!=0) {
		
		getCursorLoc(x, y, z, flags);
		if (starting) {
			x1 = x;
			y1 = y;
			starting = false;
		} else {
			x2 = x;
			y2 = y;
		}
	}
	
	logOpened = true;
	print("end: " + x1 + " " +  y1 + " " +  x2 + " " +  y2);

	if (x1 == x2 && y1 == y2) {
		makePoint(x1, y1);
	} else {
	// draw a line arrow
		makeArrow(x1, y1, x2, y2, "filled");
		
	}
	Roi.setStrokeWidth(2);
	Roi.setStrokeColor("#fafafa");
	roiManager("Add");
	roiManager("Show All");

	
	// add to results
	n = nResults;
	setResult("X1",n,x1);
	setResult("Y1",n,y1);
	setResult("X2",n,x2);
	setResult("Y2",n,y2);
 }

 macro "Delete last result [d]" {
 	IJ.deleteRows(nResults-1,nResults-1);
 	index = roiManager("count") - 1;
 	roiManager("select", index);
 	roiManager("delete");
 	run("Select None");
 }
 macro "Save results [s]" {
 	saveAs("Results");
 }
macro "Load results [l]" {
 	lineseparator = "\n";
	cellseparator = ",\t";
	
	// copies the whole RT to an array of lines
	lines=split(File.openAsString(""), lineseparator);
	
	// recreates the columns headers
	labels=split(lines[0], cellseparator);
	if (labels[0]==" ") {
		k=1; // it is an ImageJ Results table, skip first column
	} else {
		k=0; // it is not a Results table, load all columns
	}
	for (j=k; j<labels.length; j++)
	setResult(labels[j],0,0);
	
	// dispatches the data into the new RT
	run("Clear Results");
	for (i=1; i<lines.length; i++) {
		items=split(lines[i], cellseparator);
		for (j=k; j<items.length; j++)
	   		setResult(labels[j],i-1,items[j]);
	   	x1 = items[k];
	   	y1 = items[k+1];
	   	x2 = items[k+2];
	   	y2 = items[k+3];
		if (x1 == x2 && y1 == y2) {
			makePoint(x1, y1);
		} else {
		// draw a line arrow
			makeArrow(x1, y1, x2, y2, "filled");
		}
		Roi.setStrokeWidth(2);
		Roi.setStrokeColor("#fafafa");
		roiManager("Add");
		roiManager("Show All");
		   	
	}
	updateResults();
 }
 }
 macro "align_patch[p]" {
	run("Conversions...", " ");
	setTool("multipoint");
    waitForUser("Please select the alignment channel and 3 points on the REFERENCE image. Click OK when done")
    refname = getTitle();
    align_channel_ref = getSliceNumber();
    w = getWidth();
    h = getHeight();
    run("Clear Results");
    run("Measure");
    if (nResults == 3) {
    	tpx0 = getResult("X",0);
		tpy0 = getResult("Y",0);
		tpx1 = getResult("X",1);
		tpy1 = getResult("Y",1);
		tpx2 = getResult("X",2);
		tpy2 = getResult("Y",2);
    } else {
    	exit("Wrong Number of Points selected in REFERENCE image!");
    }
    setTool("multipoint");
	waitForUser("Please select the alignment channel and 3 points on the SOURCE composite image. Click OK when done");
    sourcename = getTitle();
    n_channels = nSlices;
    align_channel_src = getSliceNumber();
    run("Clear Results");
    run("Measure");
    if (nResults == 3) {
    	spx0 = getResult("X",0);
		spy0 = getResult("Y",0);
		spx1 = getResult("X",1);
		spy1 = getResult("Y",1);
		spx2 = getResult("X",2);
		spy2 = getResult("Y",2);
    } else {
    	exit("Wrong Number of Points selected in SOURCE stack!");
    }
    turboregstring = " " + w + " " + h +" -affine "+spx0+" "+spy0+" "+tpx0+" "+tpy0+" "+spx1+" "+spy1+" "+tpx1+" "+tpy1+" "+spx2+" "+spy2+" "+tpx2+" "+tpy2+" -showOutput";
    commandstring = "  title=outputstack" ;
    for (i = 1; i <= n_channels; i++) {
    	selectWindow(sourcename);
    	setSlice(i);
    	run("Duplicate...", " ");
    	rename("input" + i);
    	run("8-bit");
    	run("TurboReg ", "-transform -window input"+i+turboregstring);
    	selectWindow("input" + i);
    	run("Close");
    	selectWindow("Output");
    	setSlice(2);
    	run("Delete Slice");
    	run("8-bit");
    	rename("output"+i);
		commandstring = commandstring + " image" + i + "=output" + i;
		}
   	commandstring = commandstring + " image" + (n_channels + 1) + "=[-- None --]";
   	print(commandstring);
   	run("Concatenate...", commandstring);
   	selectWindow("outputstack");
    setSlice(align_channel_src);
    run("Duplicate...", " ");
    rename("aligned");
    selectWindow(refname);
    setSlice(align_channel_ref);
    run("Duplicate...", " ");
    rename("reference");
    run("Merge Channels...", "c1=aligned c2=reference create ignore");
    selectWindow("Composite");
    rename("overlay");
    f = File.open("");
    print(f,turboregstring);
    File.close(f);
}