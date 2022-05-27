### Script for simulating data with Bayesian Networks

### Make sure you run this in R 3.6.3

### Functions:
installPackages <- function(){
  install.packages("bnlearn")
  install.packages("visNetwork")
  install.packages("DAAG")
  install.packages("lattice")
  install.packages("tidyverse")
  install.packages("graphviz")
  install.packages("ggplot2")
}
plotNetwork <- function(nw, ht = "400px", cols = "darkgreen", labels = nodes(structure)){
  #source: http://gradientdescending.com/simulating-data-with-bayesian-networks/
  nw_as_string <- modelstring(nw)
  structure <- model2network(nw_as_string)
  if(is.null(labels)) labels <- rep("", length(nodes(structure)))
  nodes <- data.frame(id = nodes(structure),
                      label = labels,
                      color = cols,
                      shadow = TRUE
  )
  
  edges <- data.frame(from = structure$arcs[,1],
                      to = structure$arcs[,2],
                      arrows = "to",
                      smooth = FALSE,
                      shadow = TRUE,
                      color = "black")
  
  return(visNetwork(nodes, edges, height = ht, width = "100%"))
}
binaryToContinuous_2modes <- function(data, colx, coly){
  # input: dataframe, column names & column value which should be mapped to positive mode (1)
  # output: dataframe with colx & coly boolean values replaced by sample from GMM
  
  # add x&y column with all zeros:
  data[["x"]] <- I(rep(0, nrow(data)))
  data[["y"]] <- I(rep(0, nrow(data)))
  
  # iterate over rows
  for (row in 1:nrow(data)){
    # extract mode of GMM from which to sample:
    x_val <- data[row, colx]
    if (x_val == colx_pos){
      mean_x = mean_GMM
    }
    else{
      mean_x = -mean_GMM
    }
    y_val <- data[row, coly]
    if (y_val == coly_pos){
      mean_y = mean_GMM
    }
    else{
      mean_y = -mean_GMM
    }
    # sample from this mode:
    x = rnorm(1, mean=mean_x, sd=sd_GMM)
    y = rnorm(1, mean=mean_y, sd=sd_GMM)
    # add values to sim_data:
    data[row,"x"]=round(x, 3)
    data[row,"y"]=round(y, 3)
    #print(paste("Link boolean values to modes:", x_val, mean_x, y_val, mean_y))
  }
  # delete boolean cols:
  data = data[ , -which(names(data) %in% c(colx, coly))]
  # rename x & y to original col names:
  colnames(data)[which(names(data) == "x")] <- colx
  colnames(data)[which(names(data) == "y")] <- coly
  return(data)
}
binaryToContinuous_3modes <- function(data, colx, coly){
  # input: dataframe, column names which should be made numerical
  # output: dataframe with colx & coly boolean values replaced by sample from GMM
  
  # add x&y column with all zeros:
  data[["x"]] <- I(rep(0, nrow(data)))
  data[["y"]] <- I(rep(0, nrow(data)))
  
  # iterate over rows
  for (row in 1:nrow(data)){
    # extract mode of GMM from which to sample:
    x_val <- data[row, colx]
    if (x_val == "HIGH"){
      mean_x = mean_GMM
    }
    else if (x_val == "AVG"){
      mean_x = 0
    }
    else{
      mean_x = -mean_GMM
    }
    y_val <- data[row, coly]
    if (y_val ==  "HIGH"){
      mean_y = mean_GMM
    }
    else if (y_val == "AVG"){
      mean_y = 0
    }
    else{
      mean_y = -mean_GMM
    }
    # sample from this mode:
    x = rnorm(1, mean=mean_x, sd=sd_GMM)
    y = rnorm(1, mean=mean_y, sd=sd_GMM)
    # add values to sim_data:
    data[row,"x"]=round(x, 3)
    data[row,"y"]=round(y, 3)
  }
  # delete boolean cols:
  data = data[ , -which(names(data) %in% c(colx, coly))]
  # rename x & y to original col names:
  colnames(data)[which(names(data) == "x")] <- colx
  colnames(data)[which(names(data) == "y")] <- coly
  return(data)
}
scatterPlot <- function(data, colx, coly){
  # input: dataframe, column for x axis & column for y axis
  # output: ggplt scatterplot
  plt <- ggplot(data, aes_string(x=colx, y=coly)) + geom_point(alpha=0.3, color="darkgreen", size=2) + 
    coord_fixed(ratio = 1) + theme_minimal() + labs(title="Scatterplot of GMM variables")
  return(plt)
}
testGMMmodes <- function(mean, sd, n_per_mode){
  # Plot 2DGMM with modes (+/- mean, +/- mean):
  mat = matrix(ncol=2, nrow=0)
  df = data.frame(mat)
  colnames(df) <- c("x", "y")
  sd = 1
  for (i in c(-mean, mean)){
    mean_x = i
    for (j in c(-mean, mean)){
      mean_y = j
      # take 10 samples from this mode:
      for (i in 1:n_per_mode){
        x = rnorm(1, mean=mean_x, sd=sd)
        y = rnorm(1, mean=mean_y, sd=sd)
        df[nrow(df) + 1,] = c(x,y)
      }
    }
  }
  plt <- scatterPlot(df, "x", "y")
  #  ### Plot 2D GMM as scatterplot:
  #  plt <- ggplot(df, aes(x=x, y=y)) + geom_point()
  #  plt + coord_fixed(ratio = 1)
  return(plt)
}


### Load packages:
library(bnlearn)
packageVersion("bnlearn")
library(visNetwork)
#library(DAAG)
#library(tidyverse)
#install.packages("network")
#library(graphviz)
library(ggplot2)

### Check or set current working directory:
getwd()
#setwd("../../Documents/Programming/Thesis code/BN_Datasets")

### Select, load & plot network:
nw_name = "sachs" # "asia" #"cancer" #  
nw = readRDS(paste(nw_name, ".rds", sep=""))

plotNetwork(nw)

### Sample from Bayesian Network:
n_samples = 50000
sim_data = rbn(nw, n_samples)
sim_data[1:20,]

### For Sachs data: make 1st col = Akt into binary target
makeBinary <- function(data, col_idx){
  # input data & index of col for which AVG values should be changed to HIGH
  # count # samples with Akt=AVG:
  sum(data[,col_idx]=="AVG") # 15500
  sum(data[,col_idx]=="HIGH") # 3900
  sum(data[,col_idx]=="LOW") # 30000
  # transform all "AVG" values to "HIGH" value:
  data[data[,col_idx]=="AVG",col_idx]="HIGH"
  return(data)
}
# transform all "AVG" values to "HIGH" value for Akt col (index=1):
sim_data = makeBinary(sim_data, 1)
sim_data[1:20,]

### Add GMM component:

# for 2 modes mean=5; for 3 modes mean=8:
mean_GMM = 12 # 5 
sd_GMM = 2

# Plot 2D GMM to see if mean & sd are okay:
testGMMmodes(mean_GMM, sd_GMM, 10000)


# For GMM with 2 modes make Jnk, Raf, PIP2 & PIP3 columns binary (map AVG to HIGH):
sim_data = makeBinary(sim_data, 3)
sim_data = makeBinary(sim_data, 11)
sim_data = makeBinary(sim_data, 5)
sim_data = makeBinary(sim_data, 6)

colx = "Jnk" #"lung" #  "Xray" # 
colx_pos = "HIGH" # "yes" # "positive" # 
coly = "Raf" # "bronc" # "Dyspnoea" #  
coly_pos =  "HIGH" # "yes" # "True" # 

sim_data <- binaryToContinuous_2modes(sim_data, colx, colx_pos, coly, coly_pos)
sim_data <- binaryToContinuous_3modes(sim_data, colx, coly)
sim_data[1:5,]
scatterPlot(sim_data, colx, coly)

colx="PIP2"
coly="PIP3"
sim_data <- binaryToContinuous_2modes(sim_data, colx, colx_pos, coly, coly_pos)
sim_data <- binaryToContinuous_3modes(sim_data, colx, coly)
sim_data[1:5,]
scatterPlot(sim_data, colx, coly)

# save to csv:
write.csv(sim_data, paste(nw_name, "2_cat.csv", sep=""), row.names = FALSE)


