
import LinearAlgebra
using LinearAlgebra
using SparseArrays
using Random

using jInv
export prepareSyntheticPointCloudData
export subv2ind,ind2subv
function subv2ind(shape, cindices)
    """Return linear indices given vector of cartesian indices.
    shape: d-tuple of dimensions.
    cindices: n-iterable of d-tuples, each containing cartesian indices.
    Returns: n-vector of linear indices.

    Based on:
    https://discourse.julialang.org/t/psa-replacement-of-ind2sub-sub2ind-in-julia-0-7/14666/8
    Similar to this matlab function:
    https://github.com/tminka/lightspeed/blob/master/subv2ind.m
    """
    lndx = LinearIndices(Dims(shape))
    n = length(cindices)
    out = Array{Int}(undef, n)
    for i = 1:n
        out[i] = lndx[cindices[i]...]
    end
    return out
end


function ind2subv(shape, indices)
    """Map linear indices to cartesian.
    shape: d-tuple with size of each dimension.
    indices: n-iterable with linear indices.
    Returns: n-vector of d-tuples with cartesian indices.

    Based on:
    https://discourse.julialang.org/t/psa-replacement-of-ind2sub-sub2ind-in-julia-0-7/14666/8
    Similar to this matlab function:
    https://github.com/probml/pmtk3/blob/master/matlabTools/util/ind2subv.m
    """
    n = length(indices)
    d = length(shape)
    cndx = CartesianIndices(Dims(shape))
    out = Array{Tuple}(undef, n)
    for i=1:n
        lndx = indices[i]
        out[i] = cndx[lndx]
    end
    return out
end


function ddx(n)
# D = ddx(n), 1D derivative operator
	I,J,V = SparseArrays.spdiagm_internal(0 => fill(-1.0,n), 1 => fill(1.0,n)) 
	return sparse(I, J, V, n, n+1)
end


function loc2cs3D(loc1::Union{Int64,Array{Int64}},loc2::Union{Int64,Array{Int64}},loc3::Union{Int64,Array{Int64}},n::Array{Int64,1})
@inbounds cs = loc1 .+ (loc2.-1)*n[1] .+ (loc3.-1)*n[1]*n[2];
return cs;
end

function prepareSyntheticPointCloudData(P,N,npc::Int64,theta_phi_PC::Array{Float64,2} ,b::Array{Float64,2}, noiseTrans :: Float64, noiseAngle :: Float64 ,filename::String)
eps = 0.01171875*2.5 #0.0234375;
println("size of p:",size(P));
println("size of n:",size(N))
subsForward = P + (eps .*N);
subsBackwad = P - (eps .*N);
subs = P;
margin = 0.5;
Parray = Array{Array{Float64}}(undef,npc);
global d = [];
for i=1:npc
	#cursubs = subs[round(Int,(i-1)*(1/npc - margin)*length(subs))+1:min(length(subs),round(Int,i*(1/npc + margin)*length(subs)))];
	#cursubsfwd = subsForward[round(Int,(i-1)*(1/npc - margin)*length(subsForward))+1:min(length(subsForward),round(Int,i*(1/npc + margin)*length(subsForward)))];
	#cursubsbwd = subsBackwad[round(Int,(i-1)*(1/npc - margin)*length(subsBackwad))+1:min(length(subsBackwad),round(Int,i*(1/npc + margin)*length(subsBackwad)))];
	cursubs = P;
	println("size of cursubs:",size(cursubs))
	cursubsfwd = subsForward;
	cursubsbwd = subsBackwad;
	P = [cursubsbwd; cursubs; cursubsfwd];
	P = [cursubsbwd;cursubsfwd];
	
	println("size of P:",size(P))
	Parray[i] = P;
	vals = [ones(size(cursubsbwd,1));0.5.*ones(size(cursubs,1)); zeros(size(cursubsfwd,1))];
	vals = [ones(size(cursubsbwd,1)); zeros(size(cursubsfwd,1))];
	d = [d ;vals];
end

d = Array{Float64,1}(d);
println("size of d:",size(d))
#Add noise:
b = b + noiseTrans*randn(size(b,1),3)
theta_phi_PC = theta_phi_PC +  noiseAngle*randn(size(theta_phi_PC));
					
file = matopen(string(filename,"_data.mat"), "w");
write(file,"Data",d);
write(file,"P",Parray);
write(file,"Normals",d);
write(file,"margin",margin);
write(file,"b",b);
write(file,"theta_phi_PC",theta_phi_PC)
write(file,"npc",npc)
close(file);
return;
end


export readPointCloudDataFile
function readPointCloudDataFile(filename::String)
file = matopen(string(filename,"_data.mat"), "r");
Data = read(file,"Data");
P = read(file,"P");
Normals = read(file,"Normals");
Margin = read(file,"margin");
b = read(file,"b");
theta_phi_PC = read(file,"theta_phi_PC");
npc = read(file,"npc")
close(file);
return Data,P,Normals,Margin,b,theta_phi_PC,npc;
end