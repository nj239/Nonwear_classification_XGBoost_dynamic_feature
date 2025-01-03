library(data.table)#faster way to read csv 
library(parallel)# parallel computation 
library(foreach)# parallel computation 
library(doParallel )# parallel computation 
library(tsfeatures)
library(rsample)
library(runner)
library(zoo)
library(dplyr)
library(lubridate)
library(factoextra)
library(e1071)
library(vars)#Information criteria and FPE for different VAR(p)
library(caret)


Dynamic_feature<-function(dt){
  #####fill in missing data using zoo na.approx for "activity"
  dt$Activity=na.approx(dt$Activity)
  
  
  
  
  fea_name<-c("entropy","max_level_shift","max_var_shift","max_kl_shift","hurst","spike",
              "arch_acf","garch_acf","ARCH.LM","std1st_der")
  tep<-c()
  ##for  window size:121
  for (i in 1:1){
    tep1<-c(121)
    tep<-c(tep,paste0(fea_name,"_",tep1[i]))
  }
  fea_name<-tep
  rm(tep1,tep)
  #n_lag_max<-0
  
  
  n_lag=1
  
  fea_table_final<-data.frame(matrix(data = NA,ncol=1+length(fea_name)+n_lag,nrow=1))
  colnames(fea_table_final)<-c("Value",fea_name,paste0(paste0("lag",seq(120,120,120/n_lag))))
  
  fea_table<-data.frame(matrix(data = NA,ncol=1+length(fea_name)+n_lag,nrow=nrow(dt)))
  colnames(fea_table)<-c("Value",fea_name,paste0(paste0("lag",seq(120,120,120/n_lag))))
  fea_table$Value<-dt$Activity
  
  circle<-c(121)
  for (i in 1:length(circle)){
    tep<-rollapply(fea_table$Value, width = circle[i],
                   FUN = function(x) ts(x)%>%entropy() )
    fea_table[(circle[i]%/%2+1):((nrow(fea_table)-(circle[i]%/%2))),which(colnames(fea_table)==paste0("entropy_",circle[i]))]<-tep
  }
  
  #max_level_shift.
  for (i in 1:length(circle)){
    tep<-rollapply(fea_table$Value, width = circle[i],
                   FUN = function(x) ts(x)%>%max_level_shift()%>%t() %>% as.data.frame()%>%dplyr:: select(max_level_shift)%>%as.numeric() )
    fea_table[(circle[i]%/%2+1):((nrow(fea_table)-(circle[i]%/%2))),which(colnames(fea_table)==paste0("max_level_shift_",circle[i]))]<-tep
  }
  #max_var_shift
  for (i in 1:length(circle)){
    tep<-rollapply(fea_table$Value, width = circle[i],
                   FUN = function(x) ts(x)%>%max_var_shift()%>%t() %>% as.data.frame()%>%dplyr:: select(max_var_shift)%>%as.numeric() )
    fea_table[(circle[i]%/%2+1):((nrow(fea_table)-(circle[i]%/%2))),which(colnames(fea_table)==paste0("max_var_shift_",circle[i]))]<-tep
  }
  
  #max_kl_shift
  for (i in 1:length(circle)){
    tep<-rollapply(fea_table$Value, width = circle[i],
                   FUN = function(x) ts(x)%>%max_kl_shift()%>%t() %>% as.data.frame()%>%dplyr:: select(max_kl_shift)%>%as.numeric() )
    fea_table[(circle[i]%/%2+1):((nrow(fea_table)-(circle[i]%/%2))),which(colnames(fea_table)==paste0("max_kl_shift_",circle[i]))]<-tep
  }
  
  #hurst
  for (i in 1:length(circle)){
    tep<-rollapply(fea_table$Value, width = circle[i],
                   FUN = function(x) ts(x)%>%hurst()%>%as.numeric() )
    fea_table[(circle[i]%/%2+1):((nrow(fea_table)-(circle[i]%/%2))),which(colnames(fea_table)==paste0("hurst_",circle[i]))]<-tep
  }
  
  #spike
  for (i in 1:length(circle)){
    tep<-rollapply(fea_table$Value, width = circle[i],
                   FUN = function(x) ts(x)%>%stl_features()%>%t() %>% as.data.frame()%>%dplyr:: select(spike)%>%as.numeric() )
    fea_table[(circle[i]%/%2+1):((nrow(fea_table)-(circle[i]%/%2))),which(colnames(fea_table)==paste0("spike_",circle[i]))]<-tep
    
  }
  
  #arch_acf/garch_acf_/arch_r2_/garch_r2_
  for (i in 1:length(circle)){
    tep<-rollapply(fea_table$Value, width = circle[i],
                   FUN = function(x) {if (var(x,na.rm = TRUE)>0){
                     ts(x)%>%heterogeneity()%>%t() %>% as.data.frame()%>%as.numeric() }else{NA}})
    
    fea_table[(circle[i]%/%2+1):((nrow(fea_table)-(circle[i]%/%2))),which(colnames(fea_table)==paste0("arch_acf_",circle[i]))]<-tep[,1]
    fea_table[(circle[i]%/%2+1):((nrow(fea_table)-(circle[i]%/%2))),which(colnames(fea_table)==paste0("garch_acf_",circle[i]))]<-tep[,2]
    
  }
  
  #ARCH.LM
  for (i in 1:length(circle)){
    tep<-rollapply(fea_table$Value, width = circle[i],
                   FUN = function(x) ts(x)%>%arch_stat()%>%as.numeric() )
    fea_table[(circle[i]%/%2+1):((nrow(fea_table)-(circle[i]%/%2))),which(colnames(fea_table)==paste0("ARCH.LM_",circle[i]))]<-tep
  }
  
  #"std1st_der"
  for (i in 1:length(circle)){
    tep<-rollapply(fea_table$Value, width = circle[i],
                   FUN = function(x) ts(x)%>%std1st_der()%>%as.numeric() )
    fea_table[(circle[i]%/%2+1):((nrow(fea_table)-(circle[i]%/%2))),which(colnames(fea_table)==paste0("std1st_der_",circle[i]))]<-tep
  }
  
  #lag
  for (i in 1:n_lag){
    
    fea_table[,which(colnames(fea_table)==paste0("lag",i*120))]<-lag(fea_table$Value,120*i)
  }
  
  fea_table_final<-rbind(fea_table_final,fea_table)
  
  fea_table_final<-fea_table_final[-1,]
  fea_table_final<-cbind(fea_table_final[,-ncol(fea_table_final)],dt[,-4])
  return(fea_table_final)
}


library(future)
plan(multisession(workers=6))


files_path<-"file_path/Train/"

all_train_files<-list.files(files_path)

for(f in (all_train_files)){
  file_path<-paste0(files_path,f)
  if(file.exists(paste0("D:/","dynamic_features_",f))){
    print(paste0("D:/","dynamic_features_",f," exist, skipped"))
    next
  }
  print(file_path)
  future({
    result<-Dynamic_feature(read.csv(file_path))
    write.csv(result,paste0("D:/","dynamic_features_",f))
  })
  
}
