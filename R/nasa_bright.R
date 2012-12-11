library(ReadImages)
library(reshape)
library(ggplot2)

## read image - image matrix
nasa <- read.jpeg('nasa.jpg')
dim(nasa)
## reshape data by melting
nasa <- melt(nasa)
dim(nasa)
## get the RGB image - coordinates in X1, X2 and r, g, b values in 1, 2, 3
nasa <- reshape(nasa, timevar = 'X3', idvar = c('X1', 'X2'), direction = 'wide')
nasa$X1 <- -nasa$X1
dim(nasa)
## look at the region of sand
sand <- with(nasa, nasa[X2 > 1000 & X1 > -300, ])
dim(sand)
## find typical sand value, 3:5 r g b values
sand.mean <- apply(sand[, 3:5], 2, mean)
sand.sd <- apply(sand[, 3:5], 2, sd)
print(sand.mean)
print(sand.sd)
## how much do all colors deviate from sand colors?
colorz <- sweep(nasa[, 3:5], 2, sand.mean, "-")
colorz <- sweep(colorz, 2, sand.sd, '/')
plot(density(rowSums(colorz)))
## plot a binarized image, based on color z-scores
with(nasa, plot(X2, X1, col = rgb(colorz>4), asp = 1, pch = '.'))
