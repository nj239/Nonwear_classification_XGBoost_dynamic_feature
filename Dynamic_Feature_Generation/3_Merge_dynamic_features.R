files_path_60<-"D:/" #folder path
files_path_20<-"D:/" #folder path


all_train_files_60<-list.files(files_path_60)
#all_train_files_20<-list.files(files_path_20)

for(f in (all_train_files_60)){
  file_path_60<-paste0(files_path_60,f)
  t=gsub("dynamic_features_", "", f)
  if(file.exists(paste0(files_path_20,"dynamic_features_20_",t))){
    dt_60<-read.csv(file_path_60)
    dt_20<-read.csv(paste0(files_path_20,"dynamic_features_20_",t))
    dt_60<-dt_60[, 3:12]
    dt_20_60<-cbind(dt_20,dt_60)
    write.csv(dt_20_60,paste0("D:/","dynamic_features_20_60_",f))
  }
}