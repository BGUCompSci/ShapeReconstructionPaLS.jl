# module PointCloud
	
# using jInv.Mesh
# using jInv.Utils
# using jInv.InverseSolve
# using ShapeReconstructionPaLS.Utils
# using ShapeReconstructionPaLS.ParamLevelSet
# using MAT
# using SparseArrays
# using Distributed
# import jInv.ForwardShare.getData
# import jInv.ForwardShare.getSensTMatVec
# import jInv.ForwardShare.getSensMatVec
# import jInv.ForwardShare.ForwardProbType



# ## The rotation planes are: 
# # first rotation (theta) is for planes x1 and x2
# # second rotation (phi) is for planes x1 and x3
# # Sampling is done by summing planes of x1 and x2 for all values of x3.

# export PointCloudParam, getPointCloudParam
# mutable struct PointCloudParam <: ForwardProbType
    # Mesh      				:: RegularMesh
	# P						:: Array{Int64,1}
	# Normals					:: Array{Float64,2}
	# ndipsAll				:: Int64
	# workerSubIdxs			
    # #theta_phi_rad			:: Array{Float64,2}
	# #b						:: Array{Float64,2}
	# samplingBinning			:: Int64
	# method					:: String
	# Jacobian				:: SparseMatrixCSC{Float32,Int32}
	# #S						:: SparseMatrixCSC{Float32,Int32}
# end


# function getPointCloudParamInternal(Mesh::RegularMesh,P::Array{Int64,1},Normals::Array{Float64,2},ndipsAll::Int64,workerSubIdxs,samplingBinning::Int64,method)
	# return PointCloudParam(Mesh, P, Normals,ndipsAll, workerSubIdxs, samplingBinning, method,spzeros(Float32,Int32,0,0));
# end


# function getPointCloudParam(Mesh::RegularMesh,P::Array{Int64,1},Normals::Array{Float64,2},samplingBinning::Int64,numWorkers::Int64,method = MATBased)
	# ## This function does use the parallel mechanism of jInv (i.e., returns a RemoteChannel), even if numWorkers=1.	
	# if numWorkers > nworkers()
		# numWorkers = nworkers();
	# end
	# #SourcesSubInd = Array{Array{Int64,1}}(numWorkers);
	# SourcesSubInd = Array{Array{Int64}}(undef,numWorkers);
	# ActualWorkers = workers();
	# if numWorkers < nworkers()
		# ActualWorkers = ActualWorkers[1:numWorkers];
	# end
	# #pFor   = Array{RemoteChannel}(numWorkers)
	# pFor   = Array{RemoteChannel}(undef,numWorkers);
	# i = 1; nextidx() = (idx=i; i+=1; idx)
	# idx=i;
	# ndips  = 1;
	# # send out jobs
	# @sync begin
		# for w = ActualWorkers
			# @async begin
				# while true
					# idx = nextidx();
					# if idx > numWorkers
						# break
					# end
					# I_p = getIndicesOfKthWorker(numWorkers,idx,ndips);
					# # println("Sending ",collect(I_p)," to worker ", w);
					# # find src and rec on mesh
					# #SEE IF CORRECT!!!!!!!!!!!!!!!!
					# tmp = initRemoteChannel(getPointCloudParamInternal,w, Mesh,P,Normals,ndips,I_p,samplingBinning,method);
					# pFor[1]  = initRemoteChannel(getPointCloudParamInternal,w, Mesh,P,Normals,ndips,I_p,samplingBinning,method);
					# wait(pFor[1]);
					
				# end
			# end
		# end
	# end
	# return pFor # Array of Remote Refs
# end

# import jInv.Utils.clear!
# function clear!(pFor::PointCloudParam)
	# pFor.SampleMat = spzeros(0);
	# clear!(pFor.Mesh);
	# return pFor;
# end


# #include("generateDipMatrix.jl");
# #include("prepareSyntheticDipData.jl");
# include("getData.jl");
# include("sensitivityFuncs.jl");
# end