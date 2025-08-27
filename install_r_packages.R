# Install required R packages for ETIA
.libPaths(c("C:/Users/hp/Documents/R/win-library/4.5", .libPaths()))

# Install dependencies first
cat("Installing dependencies...\n")
install.packages(c("ordinal", "quantreg", "lme4", "foreach", "doParallel", 
                   "relations", "Rfast", "visNetwork", "energy", "geepack",
                   "knitr", "dplyr", "bigmemory", "coxme", "Rfast2", "Hmisc"), 
                repos = "https://cran.r-project.org/")

# Install daggity
cat("Installing daggity...\n")
install.packages("daggity", repos = "https://cran.r-project.org/")

# Install MXM from archive
cat("Installing MXM...\n")
install.packages("https://cran.r-project.org/src/contrib/Archive/MXM/MXM_1.5.5.tar.gz", 
                type = "source", repos = NULL)

cat("R packages installation completed.\n")
