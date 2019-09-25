using PyPlot;
using MAT;
using jInv.Mesh
using jInvVis
file = matopen("hand.mat")
B = read(file, "B");
close(file);
plotModel(convert(Array{Float64,3},B));
