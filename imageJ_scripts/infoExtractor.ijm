// "BatchProcessFolders"

   dir = getDirectory("Choose the original images directory ");
   
   //setBatchMode(true);
   count = 0;
   countFiles(dir);
   n = 0;
   processFiles(dir);
   //print(count+" files processed");
   
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
	if (endsWith(path, ".tif")) {
		run("Close All"); 
		open(path);
		idx1= lastIndexOf(path, "/");
        idx2= lastIndexOf(path, ".tif");
        imgname = substring(path, idx1+1, idx2);
	

	run("Show Info...");
	saveAs("text",dir+imgname+"_info.txt");	    
    run("Close");
       
    }
  }  
      run("Close All");
  
  