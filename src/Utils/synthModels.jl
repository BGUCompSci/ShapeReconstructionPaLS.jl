export getDiamondModel,getBallModel ;

function getDiamondModel(n::Array{Int64})
n_tup = tuple(n...);
n = collect(n_tup);
u = zeros(Float32,n_tup);
X,Y,Z = meshgrid(-div(n[1],2):div(n[1],2),-div(n[2],2):div(n[2],2),-div(n[3],2):div(n[3],2))
I = (abs.(X)+abs.(Y)+abs.(Z)) .<= div(n[1],4);
u[I] .= 1.0;
return u;
end

function getBallModel(n::Array{Int64})
n_tup = tuple(n...);
n = collect(n_tup);
u = zeros(Float32,n_tup);
X,Y,Z = meshgrid(-div(n[1],2):div(n[1],2),-div(n[2],2):div(n[2],2),-div(n[3],2):div(n[3],2))
I = (sqrt.(X.^2+Y.^2 + Z.^2)) .<= div(n[1],4);
u[I] .= 1.0;
return u;
end