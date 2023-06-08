// "BatchProcessFolders"

   dir = getDirectory("Choose the original images directory ");
   savedir= getDirectory("Choose directory to save converted images");
   setBatchMode(true);
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
       if (endsWith(path, ".oif")) {
        run("Close All");
           run("Bio-Formats Windowless Importer", "open=["+ path +"]");
           idx1= lastIndexOf(path, "/");
           idx2= lastIndexOf(path, ".oif");
           imgname = substring(path, idx1+1, idx2);
           run("Grays");
		    saveAs("Tiff", savedir+imgname+".tif");
            close();
       }
      run("Close All");
  }
