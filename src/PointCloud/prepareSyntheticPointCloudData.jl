
import LinearAlgebra
using LinearAlgebra
using SparseArrays
using jInv
export prepareSyntheticPointCloudData

function ddx(n)
# D = ddx(n), 1D derivative operator
	I,J,V = SparseArrays.spdiagm_internal(0 => fill(-1.0,n), 1 => fill(1.0,n)) 
	return sparse(I, J, V, n, n+1)
end

function prepareSyntheticPointCloudData(m,Minv::RegularMesh,ndipsAll::Int64,filename::String)
n = Minv.n;
#m = reshape(m,tuple(n...));
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

#Wf = Wf./(f2(Wf) .+ eps);

nx = Wf[:,1]; ny = Wf[:,2]; nz = Wf[:,3];

println("maximum norm:",maximum(f2(Wf)));
ind = findall(x -> x > 0.4, f2(Wf));
println("indices size :",size(ind))
I2 = collect(1:length(ind));
J2 = ind;
V2 = ones(size(ind));
#println("length indices:",length(indices),"length u:",length(u))
P = sparse(I2,J2,V2,length(ind),length(m));
nx = P*nx; ny = P*ny; nz = P*nz;
Normals = [nx ny nz];

pFor = getPointCloudParam(Minv,ind,Normals,1,1,MATFree)
Dremote,pFor = getData(m[:],pFor);
Data = arrangeRemoteCallDataIntoLocalData(Dremote);
					
file = matopen(string(filename,"_data.mat"), "w");
write(file,"domain",Minv.domain);
write(file,"n",Minv.n);
write(file,"Data",Data);
write(file,"P",ind);
write(file,"Normals",Normals);
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
close(file);
return n,domain,Data,P,Normals;
end