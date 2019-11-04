
import LinearAlgebra
using LinearAlgebra
using SparseArrays
using jInv
export prepareSyntheticPointCloudData

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

function prepareSyntheticPointCloudData(m,Minv::RegularMesh,ndipsAll::Int64,theta_phi_dip::Array{Float64,2} ,b::Array{Float64,2}, noiseTrans :: Float64, noiseAngle :: Float64 ,filename::String)
n = Minv.n;
m = m[:]
eps = 1e-4/Minv.h[1];
Af   = getFaceAverageMatrix(Minv)
A1 = Af[:,1:Int(size(Af,2)/3)]; println("A1 size:",size(A1))
A2 = Af[:,Int(size(Af,2)/3)+1:2*Int(size(Af,2)/3)];
A3 = Af[:,2*Int(size(Af,2)/3)+1:Int(size(Af,2))];

println("type of A1:",typeof(A1))
println("size of A1:",size(A1))

	
D1 = kron(sparse(1.0I, Minv.n[3], Minv.n[3]),kron(sparse(1.0I, Minv.n[2], Minv.n[2]),ddx(Minv.n[1])))
D2 = kron(sparse(1.0I, Minv.n[3], Minv.n[3]),kron(ddx(Minv.n[2]),sparse(1.0I, Minv.n[1], Minv.n[1])))
D3 = kron(ddx(Minv.n[3]),kron(sparse(1.0I, Minv.n[2], Minv.n[2]),sparse(1.0I, Minv.n[1], Minv.n[1])))

println("type of D1,:",typeof(D1'))
println("size of D1,:",size(D1'))

	
nx = A1*(D1'*m);#./ (sqrt.((A1*D1'*m).^2 .+ eps));
ny = A2*(D2'*m);#./ (sqrt.((A2*D2'*m).^2 .+ eps));
nz = A3*(D3'*m);#./ (sqrt.((A3*D3'*m).^2 .+ eps));


Wf = [nx ny nz];
f2(A) = [norm(A[i,:]) for i=1:size(A,1)]
nx = Wf[:,1]; ny = Wf[:,2]; nz = Wf[:,3];

println("maximum norm:",maximum(f2(Wf)));
ind = findall(x -> x > 0.8, f2(Wf));
println("indices size :",size(ind))

subs = ind2subv(Minv.n,ind);
subs = sort(subs);
ind = subv2ind(Minv.n,subs);
Normals = [nx ny nz];
margin = 0.2;


pFor = getPointCloudParam(Minv,ind,Normals,margin,theta_phi_dip ,b,1,1,MATFree)
println("pFor:",pFor);
Dremote,pFor = getData(m[:],pFor);
Data = arrangeRemoteCallPCIntoLocalData(Dremote);
#Add noise:
b = b + noiseTrans*randn(size(b,1),3)
theta_phi_dip = theta_phi_dip +  noiseAngle*randn(size(theta_phi_dip));
					
file = matopen(string(filename,"_data.mat"), "w");
write(file,"domain",Minv.domain);
write(file,"n",Minv.n);
write(file,"Data",Data);
write(file,"P",ind);
write(file,"Normals",Normals);
write(file,"margin",margin);
write(file,"b",b);
write(file,"theta_phi_dip",theta_phi_dip)
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
theta_phi_dip = read(file,"theta_phi_dip");
close(file);
return n,domain,Data,P,Normals,Margin,b,theta_phi_dip;
end