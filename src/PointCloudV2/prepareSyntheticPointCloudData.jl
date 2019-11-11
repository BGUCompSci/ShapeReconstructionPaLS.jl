
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

function prepareSyntheticPointCloudData(m,Minv::RegularMesh,npc::Int64,theta_phi_PC::Array{Float64,2} ,b::Array{Float64,2}, noiseTrans :: Float64, noiseAngle :: Float64 ,filename::String)
n = Minv.n;
m = m[:]
eps = 1e-4;



Af   = getFaceAverageMatrix(Minv)
A1 = Af[:,1:Int(size(Af,2)/3)]; #println("A1 size:",size(A1))
A2 = Af[:,Int(size(Af,2)/3)+1:2*Int(size(Af,2)/3)];
A3 = Af[:,2*Int(size(Af,2)/3)+1:Int(size(Af,2))];
	
D1 = kron(sparse(1.0I, Minv.n[3], Minv.n[3]),kron(sparse(1.0I, Minv.n[2], Minv.n[2]),ddx(Minv.n[1])))
D2 = kron(sparse(1.0I, Minv.n[3], Minv.n[3]),kron(ddx(Minv.n[2]),sparse(1.0I, Minv.n[1], Minv.n[1])))
D3 = kron(ddx(Minv.n[3]),kron(sparse(1.0I, Minv.n[2], Minv.n[2]),sparse(1.0I, Minv.n[1], Minv.n[1])))

nx = A1*(D1'*m);
ny = A2*(D2'*m);
nz = A3*(D3'*m);
#Wf,J = normalize([nx;ny;nz]);
#Wf = reshape(Wf,div(length(Wf),3),3);
Wf = [nx ny nz];
f2(A) = [norm(A[i,:]) for i=1:size(A,1)]


ind = findall(x -> x > 0.8, f2(Wf));
ind = ind[randperm(length(ind))[1:round(Int,0.5*length(ind))]];
Wf,J = normalize([nx;ny;nz]);
subs = ind2subv(Minv.n,ind);
subs = sort(subs,rev = true);
Wf = reshape(Wf,div(length(Wf),3),3);
println("size of Wf:",size(Wf))
subsForward = copy(subs); subsBackwad = copy(subs);
println("size subsForward:",size(subsForward));
println("size Wf[ind]:",size(Wf[ind,:]));
println("size of (0.01.*Wf[ind,:]):",size((0.01.*Wf[ind,:])));
subsMat = getindex.(subs,[1 2 3]);

subsForward = round.(Int64,subsMat + (3 .*Wf[ind,:]));
tmp = loc2cs3D(subsForward[:,1],subsForward[:,2],subsForward[:,3],Minv.n);
subsForward = ind2subv(Minv.n,tmp);


subsBackwad = round.(Int64,subsMat - (3 .*Wf[ind,:]));
tmp = loc2cs3D(subsBackwad[:,1],subsBackwad[:,2],subsBackwad[:,3],Minv.n);
subsBackwad = ind2subv(Minv.n,tmp);


ind = subv2ind(Minv.n,subs);
indForward = subv2ind(Minv.n,subsForward);
indBackward = subv2ind(Minv.n,subsBackwad);

println("size ind for:",size(indForward))
Normals = Wf;

margin = 0.5;
Parray = Array{SparseMatrixCSC{Float32,Int32}}(undef,npc);
global d = [];
for i=1:npc
	cursubs = subs[round(Int,(i-1)*(1/npc - margin)*length(subs))+1:min(length(subs),round(Int,i*(1/npc + margin)*length(subs)))];
	cursubsfwd = subsForward[round(Int,(i-1)*(1/npc - margin)*length(subsForward))+1:min(length(subsForward),round(Int,i*(1/npc + margin)*length(subsForward)))];
	cursubsbwd = subsBackwad[round(Int,(i-1)*(1/npc - margin)*length(subsBackwad))+1:min(length(subsBackwad),round(Int,i*(1/npc + margin)*length(subsBackwad)))];
	indForward = subv2ind(Minv.n,cursubsfwd);
	ind = subv2ind(Minv.n,cursubs);
	indBackward = subv2ind(Minv.n,cursubsbwd);
	I2 = collect(1:3*length(ind));
	J2 = [indForward ;ind ;indBackward];
	V2 = ones(3*length(ind));
	P = sparse(I2,J2,V2,3*length(ind),length(m));
	Parray[i] = P;
	vals = [ones(length(indForward));0.5.*ones(length(ind)); zeros(length(indBackward))];
	d = [d ; vals];
end
println("size of d:",size(d))

#pFor = getPointCloudParam(Minv,Parray,Normals,margin,theta_phi_PC ,b,1,1,MATFree)
#Dremote,pFor = getData(m[:],pFor);

#Data = arrangeRemoteCallPCIntoLocalData(Dremote);
#Add noise:
b = b + noiseTrans*randn(size(b,1),3)
theta_phi_PC = theta_phi_PC +  noiseAngle*randn(size(theta_phi_PC));
					
file = matopen(string(filename,"_data.mat"), "w");
write(file,"domain",Minv.domain);
write(file,"n",Minv.n);
write(file,"Data",d);
write(file,"P",Parray);
write(file,"Normals",d);
write(file,"margin",margin);
write(file,"b",b);
write(file,"theta_phi_PC",theta_phi_PC)
write(file,"npc",npc)
# write(file,"m_true",m);
#write(file,"pad",pad);
close(file);
return;
end


export readPointCloudDataFile
function readPointCloudDataFile(filename::String)
file = matopen(string(filename,"_data.mat"), "r");
domain = read(file,"domain");
n = read(file,"n");
Data = read(file,"Data");
P = read(file,"P");
Normals = read(file,"Normals");
Margin = read(file,"margin");
b = read(file,"b");
theta_phi_PC = read(file,"theta_phi_PC");
npc = read(file,"npc")
close(file);
return n,domain,Data,P,Normals,Margin,b,theta_phi_PC,npc;
end