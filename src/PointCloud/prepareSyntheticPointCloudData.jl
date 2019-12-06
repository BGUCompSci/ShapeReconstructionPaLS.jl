
import LinearAlgebra
using LinearAlgebra
using SparseArrays
using Random
import StatsBase
using StatsBase

using jInv
export prepareSyntheticPointCloudData
# export subv2ind,ind2subv
# function subv2ind(shape, cindices)
    # """Return linear indices given vector of cartesian indices.
    # shape: d-tuple of dimensions.
    # cindices: n-iterable of d-tuples, each containing cartesian indices.
    # Returns: n-vector of linear indices.

    # Based on:
    # https://discourse.julialang.org/t/psa-replacement-of-ind2sub-sub2ind-in-julia-0-7/14666/8
    # Similar to this matlab function:
    # https://github.com/tminka/lightspeed/blob/master/subv2ind.m
    # """
    # lndx = LinearIndices(Dims(shape))
    # n = length(cindices)
    # out = Array{Int}(undef, n)
    # for i = 1:n
        # out[i] = lndx[cindices[i]...]
    # end
    # return out
# end


# function ind2subv(shape, indices)
    # """Map linear indices to cartesian.
    # shape: d-tuple with size of each dimension.
    # indices: n-iterable with linear indices.
    # Returns: n-vector of d-tuples with cartesian indices.

    # Based on:
    # https://discourse.julialang.org/t/psa-replacement-of-ind2sub-sub2ind-in-julia-0-7/14666/8
    # Similar to this matlab function:
    # https://github.com/probml/pmtk3/blob/master/matlabTools/util/ind2subv.m
    # """
    # n = length(indices)
    # d = length(shape)
    # cndx = CartesianIndices(Dims(shape))
    # out = Array{Tuple}(undef, n)
    # for i=1:n
        # lndx = indices[i]
        # out[i] = cndx[lndx]
    # end
    # return out
# end


function loc2cs3D(loc1::Union{Int64,Array{Int64}},loc2::Union{Int64,Array{Int64}},loc3::Union{Int64,Array{Int64}},n::Array{Int64,1})
@inbounds cs = loc1 .+ (loc2.-1)*n[1] .+ (loc3.-1)*n[1]*n[2];
return cs;
end

function prepareSyntheticPointCloudData(P,Mesh::RegularMesh,npc::Int64,theta_phi_PC::Array{Float64,2} ,b::Array{Float64,2}, noiseTrans :: Float64, noiseAngle :: Float64 ,filename::String)
eps = 0.01171875*2.5 #0.0234375;
mid = (Mesh.domain[1:2:end] + Mesh.domain[2:2:end]) ./ 2.0;
#Add noise:
b = b + noiseTrans*randn(size(b,1),3)
println("noiseangle:",noiseAngle);
theta_phi_PC = theta_phi_PC +  noiseAngle*randn(size(theta_phi_PC));
println("theta phi after noise added:",theta_phi_PC);
b[1,:] .= 0.0;
theta_phi_PC[1,:] .= 0.0;

Parray = Array{Array{Float64}}(undef,npc);
Normals = Array{Array{Float64}}(undef,npc);
global d = [];
for i=1:npc
	s = 0; e = 1;
	#if(i  == 1)
	#	s = 0; e = 1;
	#else
	#	s=1; e = 0;
	#end
	sz = round(Int64,size(P,1)/2)+1;
	ws  = Weights( vec([collect(LinRange(s,e,sz)) ; zeros(size(P,1)-sz,1)  ] ))
	if(i == 2)
		ws = Weights(reverse(vec([collect(LinRange(s,e,sz)) ; zeros(size(P,1)-sz,1)  ] )));
	end
	sampledPointIndices = sample(collect(1:size(P,1)), ws, round(Int64,size(P,1)/2),replace=false )
	# println("sampledPointIndices:",sampledPointIndices)
	P_curr = P[sampledPointIndices,:];
	Normals[i] = P_curr[:,4:6];
	cursubs = P_curr[:,1:3];
	writedlm(string("PC_",filename,"_",i,".txt"),cursubs);
	#println("size of cursubs:",size(cursubs))
	subsForward = cursubs + (eps .*Normals[i]);
	subsBackward = cursubs - (eps .*Normals[i]);
	curr_points = [subsBackward; cursubs; subsForward];
	
	R = getRotate3D(theta_phi_PC[i,1],theta_phi_PC[i,2]);
	curr_points = curr_points .- (mid')
	curr_points =  ( curr_points )*(R') .+ (mid') 
	curr_points .+= b[i,:]'; 
	writedlm(string("PC_rotated_",filename,"_",i,".txt"),curr_points[1:round(Int64,size(curr_points,1)/3),:]);
	
	println("size of P:",size(curr_points))
	Parray[i] = curr_points;
	vals = [ones(size(subsBackward,1));0.5.*ones(size(cursubs,1)); zeros(size(subsForward,1))];
	#vals = [ones(size(cursubsbwd,1)); zeros(size(cursubsfwd,1))];
	d = [d ;vals];
end

d = Array{Float64,1}(d);
#println("size of d:",size(d))
					
file = matopen(string(filename,"_data.mat"), "w");
write(file,"Data",d);
write(file,"P",Parray);
write(file,"Normals",Normals);
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
b = read(file,"b");
theta_phi_PC = read(file,"theta_phi_PC");
npc = read(file,"npc")
close(file);
return Data,P,Normals,b,theta_phi_PC,npc;
end
