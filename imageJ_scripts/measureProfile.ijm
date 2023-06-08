/* 
This script is used to generate staining profiles across layers of a brain structure. 
It was written specifically for the hippocampus but could be adjusted for other brain regions.
*/


dir = getDirectory("Choose the original images directory ");
save_dir = getDirectory("Now choose the directory to save results");
count = 0;
countFiles(dir);
n = 0;
processFiles(dir);
run("Close All");
   
function countFiles(dir) {
      list = getFileList(dir);
      for (i=0; i<list.length; i++) {
          if (endsWith(list[i], "/"))
              countFiles(""+dir+list[i]);
          else
              count++;
      }
}

function processFiles(dir) {
      list = getFileList(dir);
      for (i=0; i<list.length; i++) {
          if (endsWith(list[i], "/"))
              processFiles(""+dir+list[i]);
          else {
             showProgress(n++, count);
             path = dir+list[i];
             processFile(path);
          }
      }
}

function processFile(path) { 
	if (endsWith(path, ".oif")) {
		run("Close All"); 
		open(path);
		idx1= lastIndexOf(path, "/");
        idx2= lastIndexOf(path, ".oif");
        imgname = substring(path, idx1+1, idx2);
	    run("Set Scale...", "distance=0 known=0 pixel=1 unit=pixel");

		if (roiManager("count") > 0) {
			roiManager("Deselect");
			roiManager("Delete");
  			}
  			
  		run("Split Channels");
		selectImage(3);
		title = getTitle();
		setTool("line");
		roiManager("Show All");
		waitForUser("add profile lines by pressing t , then click OK");
  		//print(title);
  		// TODO: get the number of profiles from ROI manager
  		roiManager("Multi Plot");
		Plot.showValues();
    	saveAs("Results", save_dir+title+"_profiles.csv");
    	
		for (i=1; i<=2; i++) {
  			selectImage(i);
  			title = getTitle();
  			print(title);
  			roiManager("Multi Plot");
			Plot.showValues();
    		saveAs("Results", save_dir+title+"_profiles.csv");
			}
			
		selectImage(3);
		roiManager("Draw");
		roiManager("Save", save_dir+imgname+"_profiles.zip");
		roiManager("Delete");
		waitForUser("Draw lines to middle of SPyr , then click OK");
		roiManager("Measure");
		// TODO: add check to make sure that the two sets of lines have the same numbers of lines
		/* If num profiles != num zeros
		 *  	"numbers do not match. image will be reloaded. start again"
		 *  	reload image
		 */
		// TODO: add a reload image button so that you can start over
		saveAs("Results", save_dir+imgname+"_meta.csv");
		roiManager("Save",save_dir+imgname+"_zeros.zip");
		run("Close All");
		
     }  
}
       
      
  
  