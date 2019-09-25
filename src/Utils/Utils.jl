module Utils
using jInv.Mesh
using SparseArrays
using Distributed
#include("rotation3D.jl")
include("synthModels.jl")
include("jointStuff.jl")
# include("ndgrid.jl")
# export ndgrid, meshgrid

export MATBased,MATFree,RBFBasedSimple1,RBFBasedSimple2,RBF10BasedSimple1, RBF10BasedSimple2,RBFBased,RBF10Based,RBF5Based;
MATBased        = "matrix";
MATFree         = "matfree";
RBFBasedSimple1 = "RBFsSimple1";
RBFBasedSimple2 = "RBFsSimple2";
RBF10BasedSimple1 = "RBFs10Simple1";
RBF10BasedSimple2 = "RBFs10Simple2";
RBFBased        = "RBFs";
RBF10Based        = "RBFs10";
RBF5Based        = "RBFs5";


export arrangeRemoteCallDataIntoLocalData,getIndicesOfKthWorker,divideDataToWorkersData


function arrangeRemoteCallDataIntoLocalData(DremoteCall::Array{Future,1})
N = length(DremoteCall);
DdivideLocal = Array{Array{Float64,2}}(undef,N);
for k=1:N
	DdivideLocal[k] = fetch(DremoteCall[k]);
end
l = 0;
for k=1:N
	l+=size(DdivideLocal[k],2);
end
Dlocal = zeros(eltype(DdivideLocal[1]),size(DdivideLocal[1],1),l)
for k=1:N
	Dlocal[:,k:N:end] = DdivideLocal[k];
end	
return Dlocal;
end

function divideDataToWorkersData(nworkers::Int64,Dlocal)
ndips = size(Dlocal,2);
dobs 	= Array{Array{Float32}}(nworkers);
for i=1:nworkers
	I_i = getIndicesOfKthWorker(nworkers,i,ndips);
	dobs[i] = Dlocal[:,I_i];
end
return dobs;
end

function divideDirectDataToWorkersData(nworkers::Int64,Dlocal)
nslices = size(Dlocal,3);
dobs 	= Array{Array{Float32,3}}(nworkers);
for i=1:nworkers
	I_i = getDirectIndicesOfKthWorker(nworkers,i,nslices);
	dobs[i] = Dlocal[:,:,I_i];
end
return dobs;
end
	
function getIndicesOfKthWorker(nworkers::Int64,k::Int64,nsrc::Int64)
return k:nworkers:nsrc;
end

function getDirectIndicesOfKthWorker(nworkers::Int64,k::Int64,nsrc::Int64)
return k:nworkers:nsrc;
end



end
