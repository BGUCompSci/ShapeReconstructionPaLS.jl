module Utils
using jInv.Mesh
using SparseArrays
using Distributed
using LinearAlgebra
include("synthModels.jl")
include("jointStuff.jl")

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
export arrangeRemoteCallPCIntoLocalData
export getDivergenceMatrix2
export arrangeRemoteCallDataIntoLocalData,getIndicesOfKthWorker,divideDataToWorkersData, divideDirectDataToWorkersData
export dividePointCloudDataToWorkersData
export PCFun
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

function arrangeRemoteCallPCIntoLocalData(DremoteCall::Array{Future,1})
N = length(DremoteCall);
println("length of dremote:",N);
println("size of dremote[1]:",size(DremoteCall)[1])
DdivideLocal = Array{Array{Float64}}(undef,N);
for k=1:N
println("size of fetch",size(fetch(DremoteCall[k])));
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
dobs 	= Array{Array{Float32}}(undef,nworkers);
for i=1:nworkers
	I_i = getIndicesOfKthWorker(nworkers,i,ndips);
	dobs[i] = Dlocal[:,I_i];
end
return dobs;
end

function divideDirectDataToWorkersData(nworkers::Int64,Dlocal)
nslices = size(Dlocal,3);
dobs 	= Array{Array{Float32,3}}(undef,nworkers);
for i=1:nworkers
	I_i = getDirectIndicesOfKthWorker(nworkers,i,nslices);
	dobs[i] = Dlocal[:,:,I_i];
end
return dobs;
end


function dividePointCloudDataToWorkersData(nworkers::Int64,Dlocal)
nslices = size(Dlocal,1);
dobs 	= Array{Array{Float64,1}}(undef,nworkers);
for i=1:nworkers
	I_i = getDirectIndicesOfKthWorker(nworkers,i,nslices);
	dobs[i] = Dlocal;
end
return dobs;
end
	
function getIndicesOfKthWorker(nworkers::Int64,k::Int64,nsrc::Int64)
return k:nworkers:nsrc;
end

function getDirectIndicesOfKthWorker(nworkers::Int64,k::Int64,nsrc::Int64)
return k:nworkers:nsrc;
end


"""
	mis,dmis,d2mis = SSDFun(dc,dobs,Wd)

	Input:

		dc::Array   -  simulated data
		dobs::Array -  measured data
		Wd::Array   -  diagonal weighting

	Output:

		mis::Real   -  misfit, 0.5*|dc-dobs|_Wd^2
		dmis        -  gradient
		d2mis       -  diagonal of Hessian

"""
function PCFun(dc::Union{Array{Float64},Array{Float32}},dobs::Union{Array{Float64},Array{Float32}},Wd::Union{Array{Float64},Array{Float32}})
	#println("dobs size:",size(dobs));
	#println("dc size:",size(dc))
	res   = vec(dc)-vec(dobs) # predicted - observed data
	#res   = vec(dc) # predicted - observed data
	Wd    = vec(Wd)
	mis   = .5*real(dot(Wd.*res,Wd.*res))  # data misfit
	dmis  = 1*Wd.*(Wd.*res)
	d2mis = Wd.*Wd
	
	
	
	# res   = vec(dc) # predicted - observed data
	# #res   = vec(dc) # predicted - observed data
	# Wd    = vec(Wd)
	# mis   = real(dot(-1*(res),ones(size(res))))/length(res);  # data misfit
	# dmis  = ones(size(res))
	# d2mis = 0*(res)
	
	
	
	# eps = 10^-3
	# res   = log.(vec(dc) .+ eps) # predicted - observed data
	# #res   = vec(dc) # predicted - observed data
	# Wd    = vec(Wd)
	# mis   = -1*real(dot(res,ones(size(res))))
	# dmis  = -1 ./ (dc .+eps);
	
	# d2mis = 1 ./ (dc.^2 .+eps);
	
	return mis, dmis, d2mis
end # function SSDFun


function getDivergenceMatrix2(Mesh::AbstractTensorMesh;saveMat = true)
# Mesh.Div = getDivergenceMatrix(Mesh::AbstractTensorMesh) builds face-to-cc divergence operator
	if isempty(Mesh.Div)
		if Mesh.dim==2
			D1 = kron(sparse(1.0I, Mesh.n[2], Mesh.n[2]),ddx(Mesh.n[1]))
			D2 = kron(ddx(Mesh.n[2]),sparse(1.0I, Mesh.n[1], Mesh.n[1]))
			Div = [D1 D2]
		elseif Mesh.dim==3
			D1 = kron(sparse(1.0I, Mesh.n[3], Mesh.n[3]),kron(sparse(1.0I, Mesh.n[2], Mesh.n[2]),ddx(Mesh.n[1])))
			D2 = kron(sparse(1.0I, Mesh.n[3], Mesh.n[3]),kron(ddx(Mesh.n[2]),sparse(1.0I, Mesh.n[1], Mesh.n[1])))
			D3 = kron(ddx(Mesh.n[3]),kron(sparse(1.0I, Mesh.n[2], Mesh.n[2]),sparse(1.0I, Mesh.n[1], Mesh.n[1])))
			Div = [D1 D2 D3]
		end
		println("size o Div:",size(Div));
		Vi = getVolumeInv(Mesh)
		F  = getFaceArea(Mesh)
		println("size of Vi:",size(Vi)); println("size of F:",size(F));
		Div = Vi*(Div*F);
		if saveMat
			Mesh.Div = Div;
		end
		return Div;
	else
		return Mesh.Div;
	end
end




end
