# Load the libraries 
library(vars)
library(urca)
library(pcalg)
# To install pcalg library you may first need to execute the following commands:
# 	source("https://bioconductor.org/biocLite.R")
# 	biocLite("graph")
# 	biocLite("RBGL")

# Read the input data 
ip = read.csv('data.csv')

# Build a VAR model 
# Select the lag order using the Schwarz Information Criterion with a maximum lag of 10
# see ?VARSelect to find the optimal number of lags and use it as input to VAR()
lag = VARselect(ip, type = "const")$selection[3]
varModel = VAR(ip, p = lag)

# Extract the residuals from the VAR model 
# see ?residuals
res = residuals(varModel)

# Check for stationarity using the Augmented Dickey-Fuller test 
# see ?ur.df
summary(ur.df(res[,'Move']))
summary(ur.df(res[,'RPRICE']))
summary(ur.df(res[,'MPRICE']))

# Check whether the variables follow a Gaussian distribution  
# see ?ks.test
ks.test(res[,'Move'], 'pnorm')
ks.test(res[,'RPRICE'], 'pnorm')
ks.test(res[,'MPRICE'], 'pnorm')

# Write the residuals to a csv file to build causal graphs using Tetrad software
write.csv(res, file = 'residuals.csv', row.names = FALSE)

# OR Run the PC and LiNGAM algorithm in R as follows,
# see ?pc and ?LINGAM 

# PC Algorithm
n = nrow(res)
m = colnames(res)
pc.fit = pc(suffStat = list(C=cor(res), n=n), indepTest = gaussCItest, alpha=0.1, labels = m)

require(Rgraphviz)

plot(pc.fit, main="CPDAG from PC algorithm")

# LiNGAM Algorithm
lingam.fit = lingam(res)

a = dim(lingam.fit$Bpruned)

require(igraph)

edL = t(lingam.fit$Bpruned)
colnames(edL) <- V
rownames(edL) <- V

g = graph.adjacency(edL, add.rownames = TRUE)
plot(g)