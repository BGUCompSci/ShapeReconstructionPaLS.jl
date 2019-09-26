module ShapeFromSilhouette
using SparseArrays
using jInv.Mesh
using jInv.Utils
using jInv.InverseSolve
using ParamLevelSet
using ShapeReconstructionPaLS.Utils
using Distributed
using MAT

import jInv.ForwardShare.getData
import jInv.ForwardShare.getSensTMatVec
import jInv.ForwardShare.getSensMatVec
import jInv.ForwardShare.ForwardProbType


## The rotation planes are:
# first rotation (theta) is for planes x1 and x2
# second rotation (phi) is for planes x1 and x3
# Screen is set in plane x2 & x3, distanced ScreenDist from x1[end].


## We assume:
# The (0,0,0) point is the middle of the ObjMesh.
# The middle of the Screen Mesh is the point (XScreenLoc,0,0);
# Camera location is in the same coordinates.
# Camera is not located inside the object domain.


export SfSParam, getSfSParam
mutable struct SfSParam <: ForwardProbType
    ObjMesh      			:: RegularMesh
    theta_phi_rad			:: Array{Float64,2}
	b			    		:: Array{Float64,2}
	nshotsAll				:: Int64
	workerSubIdxs			:: Array{Int64,1}
	ScreenMeshX2X3			:: RegularMesh
    XScreenLoc	 			:: Float64
	CamLoc					:: Array{Float64,1}
	method					:: String
	mask					:: Array{Float64};
	SampleMatT				:: SparseMatrixCSC
	Jacobian
end



function getSfSParamInternal(ObjMesh::RegularMesh,theta_phi_rad::Array{Float64,2},b,nshotsAll,workerSubIdxs,ScreenMeshX2X3::RegularMesh,
						XScreenLoc::Float64,CamLoc::Array{Float64},method,mask)
	return SfSParam(ObjMesh,theta_phi_rad,b,nshotsAll,workerSubIdxs,ScreenMeshX2X3,XScreenLoc,CamLoc,method,mask, spzeros(0,0),spzeros(0,0));
end


function getSfSParam(ObjMesh::RegularMesh,theta_phi_rad::Array{Float64,2},b,ScreenMeshX2X3::RegularMesh,
						XScreenLoc::Float64,CamLoc::Array{Float64},method, numWorkers::Int64=1,mask=(Float64)[])
	# This function does use the parallel mechanism of jInv (i.e., returns a RemoteChannel), even if numWorkers=1.
	if numWorkers > nworkers()
		numWorkers = nworkers();
	end
	SourcesSubInd = Array{Array{Int64,1}}(undef,numWorkers);
	ActualWorkers = workers();
	if numWorkers < nworkers()
		ActualWorkers = ActualWorkers[1:numWorkers];
	end
	pFor   = Array{RemoteChannel}(undef,numWorkers)
	i = 1; nextidx() = (idx=i; i+=1; idx)

	nshots  = size(theta_phi_rad,1);
	# send out jobs
	@sync begin
		for p = ActualWorkers
			@async begin
				while true
					idx = nextidx()
					if idx > numWorkers
						break
					end
					I_p = getIndicesOfKthWorker(numWorkers,idx,nshots);
					#  println("Sending ",collect(I_p)," to worker ", p);
					pFor[idx]  = initRemoteChannel(getSfSParamInternal,p, ObjMesh,theta_phi_rad[I_p,:],b[I_p,:],nshots,I_p,ScreenMeshX2X3,XScreenLoc,CamLoc,method,mask);
					wait(pFor[idx]);
				end
			end
		end
	end
	return pFor # Array of Remote Refs
end

# import jInv.Utils.clear!
# function clear!(pFor::ProjParam)
	# pFor.SampleMat = spzeros(0);
	# clear!(pFor.Mesh);
	# return pFor;
# end
include("generate2DProjectionOperator.jl");
include("getData.jl");
include("sensFuncs.jl");
include("prepareSfSDataFiles.jl")


export coarsenByTwo
function coarsenByTwo(Dlocal,n_coarse,nShots)
Dlocal = reshape(Dlocal,2*n_coarse[1],2*n_coarse[2],nShots);
Data = Dlocal[1:2:end,1:2:end,:] + Dlocal[2:2:end,1:2:end,:] + Dlocal[1:2:end,2:2:end,:] + Dlocal[2:2:end,2:2:end,:];
Data = reshape(Data,:,nShots);
Data[Data.<=1.5] = 0.0; #Why these numbers ?
Data[Data.>=1.5] = 1.0;

return Data;
end
end
