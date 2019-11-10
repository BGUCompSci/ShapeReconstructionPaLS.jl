
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
Wf,J = normalize([nx;ny;nz]);
Wf = reshape(Wf,div(length(Wf),3),3);
f2(A) = [norm(A[i,:]) for i=1:size(A,1)]


println("maximum norm:",maximum(f2(Wf)));
ind = findall(x -> x > 0.95, f2(Wf));
println("indices size :",size(ind))
ind = ind[randperm(length(ind))[1:round(Int,0.5*length(ind))]];

subs = ind2subv(Minv.n,ind);
subs = sort(subs);
ind = subv2ind(Minv.n,subs);
Normals = Wf;
println("size of normals before choose:",size(Normals));

margin = 0.1;
npc = 2;
Parray = Array{Array{Int64,1}}(undef,npc);
global d = [];
for i=1:npc
	cursubs = subs[round(Int,(i-1)*(1/npc - margin)*length(subs))+1:min(length(subs),round(Int,i*(1/npc + margin)*length(subs)))];
	println("cursubs size:",size(cursubs));
	for ii = 1:length(cursubs)
		cursubs[ii] = round.(Int,cursubs[ii] .+ b[i]);
	end	
	ind = subv2ind(Minv.n,cursubs);
	Parray[i] = Array{Int64}(ind);
	I2 = collect(1:length(ind));
	J2 = ind;
	V2 = ones(size(ind));
	P = sparse(I2,J2,V2,length(ind),length(m));
	nxt = P*nx; nyt = P*ny; nzt = P*nz;
	curnormals = [nxt; nyt; nzt];
	d = [d;curnormals];
end


pFor = getPointCloudParam(Minv,Parray,Normals,margin,theta_phi_PC ,b,1,1,MATFree)
println("before getdata")
Dremote,pFor = getData(m[:],pFor);
println("after getdata")

Data = arrangeRemoteCallPCIntoLocalData(Dremote);
#Add noise:
b = b + noiseTrans*randn(size(b,1),3)
theta_phi_PC = theta_phi_PC +  noiseAngle*randn(size(theta_phi_PC));
					
file = matopen(string(filename,"_data.mat"), "w");
write(file,"domain",Minv.domain);
write(file,"n",Minv.n);
write(file,"Data",Data);
write(file,"P",Parray);
write(file,"Normals",Normals);
write(file,"margin",margin);
write(file,"b",b);
write(file,"theta_phi_PC",theta_phi_PC)
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
close(file);
return n,domain,Data,P,Normals,Margin,b,theta_phi_PC;
end