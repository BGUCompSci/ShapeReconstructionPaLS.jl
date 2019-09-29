using LinearAlgebra
using Statistics
using DelimitedFiles;
using jInv.Mesh;
using MAT;
using jInv
using jInv.InverseSolve
using ShapeReconstructionPaLS
using ShapeReconstructionPaLS.DipTransform;
using ParamLevelSet;
using ShapeReconstructionPaLS.ShapeFromSilhouette;
using ShapeReconstructionPaLS.Utils;
using Statistics
using Distributed
using SparseArrays
using Random

ENV["MPLBACKEND"] = "Qt4Agg"

plotting = false;

if plotting
	using PyPlot;
	using jInvVis;
	close("all");
end

####
#Parameters to choose if: inversion with dip only/visual hull only/ joint inversion
invertDip = true;
invertVis = false;
#####

#Place new added RBFs according to gradient values or randomly:
locateRBFwithGrads = true;


################################################
### Reading the model and pad ##################
################################################
model = "skull_160"; ## skull
 #model = "dancing220";
file = matopen(string(pwd(),"/models/",model,".mat"));


A = read(file, "B");
close(file);
A = convert(Array{Float32,3},A[1:1:end,1:1:end,1:1:end]);
n_inner = collect(size(A));
pad = ceil.(Int64,0.1*n_inner[1]);
n = n_inner .+ 2*pad;


n_tup = tuple(n...)
volume = prod(n);
n1 = n_tup[1];
n2 = n_tup[2];
n3 = n_tup[3];

m = zeros(Float64,n_tup);
mask = zeros(Int8,n_tup);
m[pad+1:end-pad,pad+1:end-pad,pad+1:end-pad] = A; A = 0;
mask[pad+1:end-pad,pad+1:end-pad,pad+1:end-pad] .= 1;
if plotting
	plotModel(m);
	savefig("TrueModel.png");
end



################################################################
### Preparing the synthetic data using mat-free implementation
################################################################
Mesh = getRegularMesh([0.0 5.0 0.0 5.0 0.0 5.0],n);
ScreenMesh = getRegularMesh(Mesh.domain[3:6],n[2:3]); ## for vis.
ScreenDomain = ScreenMesh.domain;
n_screen = ScreenMesh.n;


noiseAnglesDeg =0.0#4.0;
#original was mean(2*Mesh.h);
noiseTrans =0.0#mean(8*Mesh.h);
noiseAngles =deg2rad(noiseAnglesDeg);
#########################################################################################################
### Dipping Data ########################################################################################
#########################################################################################################
samplingBinning = 2;
nDips = 0;
nShots = 0;
if invertDip
	println("Data gen resolution:")
	println(n)
	noiseSample = 2.0*prod(Mesh.h);
	nDips = 100;
	theta_phi_dip = deg2rad.(convert(Array{Float64,2}, [rand(0:359,nDips)  rand(0:90,nDips)]));
	dipDataFilename = string("dipData_",model,"_dataRes",n,"_bin",samplingBinning,"_noiseSampleAnglesTrans",[noiseSample;noiseAngles;noiseTrans],"_dips",nDips);
	prepareSyntheticDipDataFiles(m,Mesh,dipDataFilename,pad,theta_phi_dip,samplingBinning, noiseSample, noiseAngles,noiseTrans,MATFree);

	#########################################################################################################
	### Read the dipping data ###############################################################################
	#########################################################################################################
	n,domain,samplingBinning,DataDip,theta_phi_dip,pad = readDipDataFile(dipDataFilename);
end


####################################


#################################################################################################################
### Generate and Read the visual data ###############################################################################
#################################################################################################################

if invertVis
	nShots = 16;
	noiseSample = 0.0;
	factorShotsCreation = 4;
	theta_phi_vis = deg2rad.(convert(Array{Float64,2}, [rand(0:359,factorShotsCreation*nShots)  rand(0:90,factorShotsCreation*nShots)]));

	

	
	VisDataFilename = string("visData_",model,"_dataRes",n,"_noiseSampleAnglesTrans",[noiseSample;noiseAngles;noiseTrans],"_shots",nShots);

	XlocScreen = Mesh.domain[2];
	LocCamera = [-16.0*(Mesh.domain[2] - Mesh.domain[1]) ; 0.0 ;0.0]; ## 0.0 here is the middle of the domain
	noiseCamLoc = 0.0;
	noiseScreenLoc = 0.0;
	prepareSyntheticVisDataFiles(m,Mesh,ScreenMesh,pad,theta_phi_vis,XlocScreen,LocCamera,
	VisDataFilename,factorShotsCreation*nShots, noiseCamLoc, noiseScreenLoc, noiseAngles,noiseTrans,MATFree);

	##################################################################################################################
	##################################################################################################################

	n,domain,XScreenLoc,CameraLoc,ScreenDomain,n_screen,DataVis,theta_phi_vis = readVisDataFile(VisDataFilename);
	norms = vec(sum(DataVis,1));
	idsorted = sortperm(norms);
	theta_phi_vis = theta_phi_vis[idsorted[(end-nShots+1):end],:];
	DataVis = DataVis[:,idsorted[(end-nShots+1):end]];
	# for kkk = 1:3:nShots
		# dk = DataVis[:,kkk];
		# dk = reshape(dk,tuple(n_screen...));
		# figure();
		# plotModel(dk)
	# end

end
ScreenMesh 	= getRegularMesh(ScreenDomain,div.(n_screen,samplingBinning));
Mesh = getRegularMesh(domain,div.(n,samplingBinning));
pad = div(pad,samplingBinning);
if samplingBinning==2 && invertVis
	DataVis = coarsenByTwo(DataVis,ScreenMesh.n,nShots);
	if plotting
		figure();
		for kkk = 1:max(4,nShots)
			dk = DataVis[:,kkk];
			dk = reshape(dk,tuple(ScreenMesh.n...));
			subplot(8,8,kkk);
			plotModel(dk);
			
		end
		savefig(string("VisTrueModel.png"));
	end
end

samplingBinning = 1;
println("Inv mesh:")
println(Mesh.n)

# method = RBFBasedSimple2;
# method = RBF10BasedSimple2;
# method = RBF10BasedSimple1
# method = RBFBasedSimple1

method = RBF10Based; methodName="RBF10Based";

#method = MATFree; methodName="MATFree"

n_tup = tuple(Mesh.n...);
nWorkers = nworkers();
misfun = SSDFun; ## least squares


if invertDip
	### Set up the dip inversion
	pForDip = getDipParam(Mesh,theta_phi_dip,zeros(nDips,3),samplingBinning,nWorkers,method);
	
	### Create Dip param if we use the GRAD of MATFREE to locate new RBFs
	if(locateRBFwithGrads)
		pForDip_MATFree = getDipParam(Mesh,theta_phi_dip,zeros(nDips,3),samplingBinning,nWorkers,MATFree);

	end
	dobsDip = divideDataToWorkersData(nWorkers,DataDip);
	Wd_Dip = Array{Array{Float32}}(undef,nWorkers);
	for k=1:length(Wd_Dip)
		Wd_Dip[k] = ones(Float32,size(dobsDip[k]));
	end

end

if invertVis
### Set up the vis inversion
	pForVis = getVisHullParam(Mesh,theta_phi_vis,zeros(nShots,3),ScreenMesh,XlocScreen,LocCamera,method,nWorkers);
	dobsVis = divideDataToWorkersData(nWorkers,DataVis);
	Wd_Vis = Array{Array{Float32}}(nWorkers);
	for k=1:length(Wd_Vis)
		println("sqrt:",0.1*(Mesh.domain[2]-Mesh.domain[1])*prod(Mesh.h[2:3]));
		Wd_Vis[k] = sqrt(0.05*(Mesh.domain[2]-Mesh.domain[1])*prod(Mesh.h[2:3]))*ones(Float32,size(dobsVis[k]));
		# Wd_Vis[k] = sqrt(0.3)*ones(Float32,size(dobsVis[k]));

	end
end

if invertVis && invertDip
	n_Moves_all = nShots + nDips;
elseif invertDip
	n_Moves_all = nDips;
else
	n_Moves_all = nShots;
end



isSimple = 0;
isRBF10 = 0;

#########################################################################################################
############### Volumetric inversion (pixel-wise) #######################################################
#########################################################################################################
if method == MATBased || method == MATFree
# #### Set active cells
	mask = zeros(Int8,n_tup);
	mask[pad+1:end-pad,pad+1:end-pad,pad+1:end-pad] .= 1;
	Iact = speye(Float32,prod(Mesh.n));
	Iact = Iact[:,mask[:] .== 1];mask = 0;
	IactPlot = Iact;
	sback = zero(Float32);

	#### USE A BOUND MODEL or bounds
	a = 0.0;
	b = 1.0;
	
	nact = size(Iact,2);
	modfun = identityMod;
	boundsHigh = zeros(Float32,nact) + b;
	boundsLow = zeros(Float32,nact) + a;
	mref = zeros(Float64,nact);
	modfunForPlotting = modfun;


	#### Use smoothness regularization
	regfun = (m, mref, M) -> wTVReg(m, mref, M,Iact=Iact);
	HesPrec=getSSORCGRegularizationPreconditioner(1.0,1e-3,100)
	# HesPrec = getEmptyRegularizationPreconditioner();
	alpha = 1e-10; #1e-10;
	m0 = copy(mref);
###############################################################################################
############## USE RBF dictionary #############################################################
###############################################################################################

else
	nRBF = 5;

	isRBF10 = (method == RBF10BasedSimple1 || method==RBF10BasedSimple2 || method==RBF10Based);
	isSimple = (method==RBFBasedSimple1 || method==RBF10BasedSimple1 || method==RBFBasedSimple2 || method==RBF10BasedSimple2);

	if isRBF10
		numParamOfRBF = 10;
	else
		numParamOfRBF = 5;
	end
	n_m_simple = numParamOfRBF*nRBF;
	nAll = isSimple ? numParamOfRBF*nRBF : numParamOfRBF*nRBF + n_Moves_all*5;

	modfun = identityMod;

	if isSimple
		modfunForPlotting = (m)->ParamLevelSetModelFunc(Mesh,m;computeJacobian = 0,
			sigma = getDefaultHeavySide() ,bf = 1, numParamOfRBF = numParamOfRBF);
	else
		modfunForPlotting = (m)->ParamLevelSetModelFunc(Mesh,m[1:(numParamOfRBF*nRBF)];computeJacobian = 0,
			sigma = getDefaultHeavySide() ,bf = 1, numParamOfRBF = numParamOfRBF);
	end

	Iact = 1.0;
	IactPlot = 1.0;
	sback = zero(Float32);
	a = -100000000.0;
	b = 10000000.0;
	boundsHigh = zeros(Float32,nAll) .+ b;
	boundsLow = zeros(Float32,nAll) .+ a;
	# m0 - initial guess. mref - reference for regularization.
	mref = zeros(Float32,nAll);
	m0   = zeros(Float64,n_m_simple);
	mCenters = zeros(Float32,n_tup);
	mref[1:numParamOfRBF:n_m_simple] = ParamLevelSet.centerHeavySide*2.0*rand(nRBF);
	if isRBF10
		# here we initialize B as an identity matrix
		mref[2:10:n_m_simple] .= 1.0;
		mref[5:10:n_m_simple] .= 1.0;
		mref[7:10:n_m_simple] .= 1.0;
		boundsLow[2:10:n_m_simple] .= 0.05;
		boundsLow[5:10:n_m_simple] .= 0.05;
		boundsLow[7:10:n_m_simple] .= 0.05;
	else
		mref[2:5:n_m_simple] .= 1.0;
		boundsLow[2:5:n_m_simple] .= 0.05;
	end
	mref[(numParamOfRBF-2):numParamOfRBF:n_m_simple] .= (Mesh.domain[1] + Mesh.domain[2])/2;
	mref[(numParamOfRBF-1):numParamOfRBF:n_m_simple] .= (Mesh.domain[3] + Mesh.domain[4])/2;
	mref[numParamOfRBF:numParamOfRBF:n_m_simple] .= (Mesh.domain[5] + Mesh.domain[6])/2;
	m0[:] = mref[1:n_m_simple];
	if !isSimple
		if invertDip
			m0 = wrapRBFparamAndRotationsTranslations(m0,theta_phi_dip,zeros(size(theta_phi_dip,1),3));
		end
		if invertVis
			m0 = wrapRBFparamAndRotationsTranslations(m0,theta_phi_vis,zeros(size(theta_phi_vis,1),3));
		end
	end
	mref[:] = m0;
	m0[(numParamOfRBF-2):numParamOfRBF:n_m_simple] .+= 0.1*(Mesh.domain[2] - Mesh.domain[1])*randn(nRBF);
	m0[(numParamOfRBF-1):numParamOfRBF:n_m_simple] .+= 0.1*(Mesh.domain[4] - Mesh.domain[3])*randn(nRBF);
	m0[numParamOfRBF:numParamOfRBF:n_m_simple]     .+= 0.1*(Mesh.domain[6] - Mesh.domain[5])*randn(nRBF);

	alpha = 5e-1;

	II = (sparse(1.0I, nAll,nAll));
	regfun = (m, mref, M)->TikhonovReg(m,mref,M,II);
	HesPrec = getEmptyRegularizationPreconditioner();

	if isRBF10
		spd_reg = (m,mref,M) -> RBF_SPD_regularization(m,mref,nRBF);
		regfun 	 		= [regfun;spd_reg];
		alpha 		 	= [alpha;(1e-3)*alpha];
		mref 			= [mref zeros(size(mref))];
	end
end

################################################################################################################
####### Set up inversion #######################################################################################
################################################################################################################

	### In case we use MATFREE gradMisfit to locate new RBFs
	if(locateRBFwithGrads)
		mask=0;
		Iact_free = 1.0;
		#### USE A BOUND MODEL or bounds
		a_free = 0.0;
		b_free= 1.0;

		pMis_Free = getMisfitParam(pForDip_MATFree, Wd_Dip, dobsDip, misfun, Iact_free,sback);
	end
if invertVis && invertDip
	if isSimple
		Wd   = [Wd_Dip;Wd_Vis];
		dobs = [dobsDip;dobsVis];
		pFor = [pForDip;pForVis];
		pMisRFs = getMisfitParam(pFor, Wd, dobs, misfun, Iact,sback);
	else
		modfun1 = (mAll) -> splitTwoRotationsTranslations(mAll,nDips,nShots,1);
		modfun2 = (mAll) -> splitTwoRotationsTranslations(mAll,nDips,nShots,2);
		pMisRFs1 = getMisfitParam(pForDip, Wd_Dip, dobsDip, misfun, Iact,sback,ones(length(pForDip)),modfun1);
		pMisRFs2 = getMisfitParam(pForVis, Wd_Vis, dobsVis, misfun, Iact,sback,ones(length(pForVis)),modfun2);
		pMisRFs = [pMisRFs1;pMisRFs2];
		pMisRFs1 = 0;
		pMisRFs2 = 0;
	end
	n_Moves_all = nShots + nDips;

	elseif invertDip
		pMisRFs = getMisfitParam(pForDip, Wd_Dip, dobsDip, misfun, Iact,sback);
	else
	pMisRFs = getMisfitParam(pForVis, Wd_Vis, dobsVis, misfun, Iact,sback);
end

cgit  = 50; # not used.
maxit = 5;
pcgTol = 1e-2; #not used

@everywhere function updateAngles(pFor::Array{RemoteChannel},theta_phi)
@sync begin
	@async begin
		for k=1:length(pFor)
			pFor[k] = remotecall_fetch(updateAngles,pFor[k].where,pFor[k],theta_phi);
		end
	end
end
return pFor;
end

@everywhere function updateAngles(pFor::RemoteChannel,theta_phi)
	pForMATFREE  = take!(pFor)
	worker_theta_phi = theta_phi[pForMATFREE.workerSubIdxs,:]
	pForMATFREE.theta_phi_rad=copy(worker_theta_phi);
	put!(pFor,pForMATFREE)
	return pFor;
end


function myDump(mc::Vector,Dc,iter,pInv,pMis,method="",ndips=0,noiseAngle=0,noiseTranslation=0,useVisual=false,nshots=4,plotFun=modfunForPlotting)
	# if resultsFilename!=""
		# Temp = splitext(resultsFilename);
		# Temp = string(Temp[1],"_GN",iter,Temp[2]);

			ntup = tuple((pInv.MInv.n)...);
			fullMc = plotFun(mc)[1];

		if plotting
			fullMc = reshape(IactPlot*fullMc[:],ntup);
			# fullMc = reshape(fullMc[:],ntup);
			close(888);
			figure(888);
			plotModel(fullMc,true,pInv.MInv);
			sleep(1.0);
			if(isempty(method))
				savefig(string("It",iter,"_",ntup,".png"));
				writedlm(string("Params_iter_",iter,".dat"),convert(Array{Float16},mc));
				writedlm(string("Volume_iter_",iter,"_",ntup,".dat"),convert(Array{Float16},fullMc));
			else
				visualStr ="noVis"
				if(useVisual)
					visualStr = "withVis"
				end
				savefig(string(model,"_Method_",method,"_",visualStr,"_nShots_",nshots,"_Ndips_",ndips,"_noiseAngle",noiseAngle,"_noiseTrans",noiseTranslation,"_It",iter,"_",ntup,".png"));
				writedlm(string(model,"_Method_",method,"_",visualStr,"_nShots_",nshots,"_Ndips_",ndips,"_noiseAngle",noiseAngle,"_noiseTrans",noiseTranslation,"_Params_iter_",iter,".dat"),convert(Array{Float16},mc));
				writedlm(string(model,"_Method_",method,"_",visualStr,"_nShots_",nshots,"_Ndips_",ndips,"_noiseAngle",noiseAngle,"_noiseTrans",noiseTranslation,"_Volume_iter_",iter,"_",ntup,".dat"),convert(Array{Float16},fullMc));
				if(iter > 1)
				rm(string(model,"_Method_",method,"_",visualStr,"_nShots_",nshots,"_Ndips_",ndips,"_noiseAngle",noiseAngle,"_noiseTrans",noiseTranslation,"_Volume_iter_",iter-1,"_",ntup,".dat"));
				end
			end
		end
	# end
end

pInv = getInverseParam(Mesh,modfun,regfun,alpha,mref,boundsLow,boundsHigh,
                     maxStep = 2e-1,pcgMaxIter=cgit,pcgTol=pcgTol,
					 minUpdate = 0.01, maxIter = maxit,HesPrec=HesPrec);

# end

#### Projected Gauss Newton
mc = m0;
Dc = nothing
if(method == MATBased || method == MATFree)
	pInv.maxIter = 10;
	global mc,Dc,flag,his = projGN(mc,pInv,pMisRFs,solveGN=projGNexplicit);
else
  pInv.maxIter = 10;
	global mc,Dc,flag,his = projGN(mc,pInv,pMisRFs,solveGN=projGNexplicit);
end
pInv.maxIter = 10;
myDump(mc,0,1,pInv,0,methodName,nDips,noiseAnglesDeg,noiseTrans,invertVis,nShots);


locateWithGradVis = false;



if !(method == MATBased || method == MATFree)
	outerIter = 40;
	grad_u = nothing;
	new_RBF_location = [];
	@sync for iterNum=2:outerIter
		println("Inversion using ",nRBF," basis functions")
		new_nRBF = 5;

		# Dc - the data using our method.

		u = modfunForPlotting(mc)[1];
		Ii = findall((u.>0.3) .& (u.<0.6));
		#Lets compute the gradient of the misfit function:
		if(locateRBFwithGrads)
			(m11,theta_phi1,b1) 	= splitRBFparamAndRotationsTranslations(mc,nRBF,nDips,numParamOfRBF)
			#println("sanity check: len of thetha phi:",size(theta_phi1));
			#theta_phi1 = theta_phi1[1:nDips,:];
			#println("sanity check: len of thetha phi updated:",size(theta_phi1));
			println("update angles");
			updateAngles(pForDip_MATFree,theta_phi1);
			pMis_Free = getMisfitParam(pForDip_MATFree, Wd_Dip, dobsDip, misfun, Iact_free,sback);
			println("computing grad misfit")

      if(locateWithGradVis)
      
      mdownsampled = zeros(size(u));
      mdownsampled=m[1:2:end,1:2:end,1:2:end]
      println("size mdown:",size(mdownsampled));
      println("size u:",size(u));
      grad_u = abs.(mdownsampled[:]-u[:]);
      grad_u = Iact_free*grad_u;
      relevantGradPoints = grad_u[Ii];
      
      else
			#sig = copy(u);
			grad_u = computeGradMisfit(u,Dc,pMis_Free)
			grad_u = abs.(Iact_free*grad_u[:]);
			relevantGradPoints = grad_u[Ii];
      end
      
		end
		if length(Ii) > new_nRBF
			if locateRBFwithGrads
				finalPoints=zeros(Int,new_nRBF);
				point=1;
				added=1;
				tries=1;
				width = 2; # determines the size of the neighborhood to ignore when adding new RBFs
				@sync while (point <=new_nRBF)
					relevantGradPoints = grad_u[Ii];
					println(point)
					added=1;

					maxVal = maximum(relevantGradPoints);
					idmax = argmax((relevantGradPoints))
					idxMaxVal = Ii[idmax];
					relevantGradPoints[idmax]=0;
					grad_u[idxMaxVal] = 0;
					#i,j,k = ind2sub(n_tup,idxMaxVal);
					(i,j,k) = Tuple(CartesianIndices(n_tup)[idxMaxVal]);
					i = max(min(i,n1-width),width+1)
					j = max(min(i,n2-width),width+1)
					k = max(min(i,n3-width),width+1)
					grad_u = reshape(grad_u,n_tup);
					grad_u[(i-width):min(i+width,n_tup[1]),(j-width):min(j+width,n_tup[2]),(k-width):min(k+width,n_tup[3])] .= 0;
					if((!(idxMaxVal in Ii ) || idxMaxVal in new_RBF_location) && (tries<length(relevantGradPoints)))
						tries = tries + 1;
						if(tries >= length(relevantGradPoints))
							grad_u = computeGradMisfit(u,Dc,pMis_Free)
							grad_u = abs.(Iact_free*grad_u[:]);
							relevantGradPoints = grad_u[Ii];
						end
						continue;
					end
					global new_RBF_location = append!(new_RBF_location,idxMaxVal);
					finalPoints[point] = idxMaxVal;
					mCenters[idxMaxVal] = 1;
					point = point + 1;
				end
				Ii = finalPoints;
				print("new RBFs added in locations:")
				println(Ii);
				print("all added RBF locations:")
				println(new_RBF_location)
			else
				Ii = Ii[randperm(length(Ii))[1:new_nRBF]];
			end
		end
		#(I1,I2,I3) = ind2sub(n_tup,I);
	
		(I1,I2,I3) = Tuple(CartesianIndices(n_tup)[Ii]);
		I1 = collect(Tuple(I1))[1];
		I2 = collect(Tuple(I2))[1];
		I3 = collect(Tuple(I3))[1];
		X1 = I1.*Mesh.h[1] .- 0.5*Mesh.h[1] .+ Mesh.domain[1];
		X2 = I2.*Mesh.h[2] .- 0.5*Mesh.h[2] .+ Mesh.domain[3];
		X3 = I3.*Mesh.h[3] .- 0.5*Mesh.h[3] .+ Mesh.domain[5];
		n_mc_old = numParamOfRBF*nRBF;
		n_mc_new = numParamOfRBF*nRBF + length(Ii)*numParamOfRBF;
		mc_new = zeros(length(mc) + length(Ii)*numParamOfRBF);
		if isRBF10
			mref = pInv.mref[:,1];
		end
		mref_new = zeros(Float32,length(mc) + length(Ii)*numParamOfRBF);

		mc_new[1:n_mc_old] = mc[1:n_mc_old];

		## Copying theta_phi and trans
		#(m11,theta_phi1,b1) 	= splitRBFparamAndRotationsTranslations(mc,nRBF,size(theta_phi_dip,1),numParamOfRBF)
		mc_new[(n_mc_new+1):end] = mc[(n_mc_old+1):end];
		mref_new[(n_mc_new+1):end] = mref[(n_mc_old+1):end];
		mc_new[(n_mc_old+1):numParamOfRBF:n_mc_new] .= 0.0; #ParamLevelSet.centerHeavySide - ParamLevelSet.deltaHeavySide;
		mc_new[(n_mc_old+numParamOfRBF-2):numParamOfRBF:n_mc_new] .= X1;
		mc_new[(n_mc_old+numParamOfRBF-1):numParamOfRBF:n_mc_new] .= X2;
		mc_new[(n_mc_old+numParamOfRBF):numParamOfRBF:n_mc_new] .= X3;
		pInv.boundsHigh = zeros(Float32,length(mc_new)) .+ b;
		boundsLow = zeros(Float32,length(mc_new)) .+ a;
		boundsLow[1:n_mc_old] = pInv.boundsLow[1:n_mc_old];
		if isRBF10
			mc_new[(n_mc_old+2):10:n_mc_new] .= 3.0;
			mc_new[(n_mc_old+5):10:n_mc_new] .= 3.0;
			mc_new[(n_mc_old+7):10:n_mc_new] .= 3.0;
			boundsLow[(n_mc_old+2):10:n_mc_new] .= 0.05;
			boundsLow[(n_mc_old+5):10:n_mc_new] .= 0.05;
			boundsLow[(n_mc_old+7):10:n_mc_new] .= 0.05;
		else
			mc_new[(n_mc_old+2):5:n_mc_new] .= 3.0;
			boundsLow[(n_mc_old+2):5:n_mc_new] .= 0.05;
		end
		pInv.boundsLow = boundsLow;
		mref_new[1:n_mc_new] = mc_new[1:n_mc_new];

		global mc = mc_new;


		II = (sparse(1.0I, length(mc_new),length(mc_new)));
		pInv.regularizer = (m, mref, M)->TikhonovReg(m,mref,M,II);
		pInv.alpha .*= 0.8;
		if isRBF10
			spd_reg = (m,mref,M) -> RBF_SPD_regularization(m,mref,nRBF);
			pInv.regularizer  = [pInv.regularizer;spd_reg];
			pInv.mref 			  = [mref_new zeros(length(mref_new))];
		else
			pInv.mref = mref_new;
		end
		
		global mc, Dc,flag,his = projGN(mc,pInv,pMisRFs,solveGN=projGNexplicit);
		myDump(mc,0,iterNum,pInv,0,methodName,nDips,noiseAnglesDeg,noiseTrans,invertVis,nShots);
		
		if iterNum==outerIter
			pInv.alpha *= 0.1;
			pInv.maxIter = 10;
			mc, = projGN(mc,pInv,pMisRFs,solveGN=projGNexplicit);
			myDump(mc,0,iterNum+1,pInv,0,methodName,nDips,noiseAnglesDeg,noiseTrans,invertVis,nShots);
		end
		global nRBF += new_nRBF;
    
	end

	
	if (method==RBFBasedSimple1 || method==RBF10BasedSimple1 || method==RBFBasedSimple2||method==RBF10BasedSimple2 ||method==RBF10Based ||method==RBFBased)
		meshForPlotting = getRegularMesh(Mesh.domain,Mesh.n*2);
		ntup = tuple((meshForPlotting.n)...);
		if isSimple
			modfunForPlotting = (m)->ParamLevelSetModelFunc(meshForPlotting,m;computeJacobian = 0,
				sigma = getDefaultHeavySide() ,bf = 1, numParamOfRBF = numParamOfRBF);
		else
			modfunForPlotting = (m)->ParamLevelSetModelFunc(meshForPlotting,m[1:(numParamOfRBF*nRBF)];computeJacobian = 0,
				sigma = getDefaultHeavySide() ,bf = 1, numParamOfRBF = numParamOfRBF);
		end
		fullMc = modfunForPlotting(mc)[1];
		writedlm(string(model,"_100Dips_64Shots_Noise0Deg","_Volume_final_",ntup,"_from_",tuple((Mesh.n)...),".dat"),convert(Array{Float16},fullMc));
		m = convert(Array{Float16},m);
		writedlm(string(model,"_100Dips_64Shots_Noise0Deg","_Volume_orig.dat"),m);
		writedlm(string(model,"_100Dips_64Shots_Noise0Deg","_RBF_centers.dat"),mCenters);
	end


# else
	# for k=1:40
		# sref = pInv.modelfun(mc)[1];
		# println(size(mc))
		# mref = copy(mc);
		# mref[sref.>0.5] =  1.0; #Ostu's method ?
		# mref[sref.<0.5] = 0;
		# sigma = spdiagm((sref-0.5).^2 + 1e-5);

		# tikhregfun = (m, mref, M)->TikhonovReg(m,mref,M,sigma);
		# if k==1
			# pInv.regularizer = tikhregfun#[pInv.regularizer,tikhregfun];
			# pInv.alpha       = 1e-5*(10^k)#[pInv.alpha;1e-5*(10^k)];
			# pInv.mref 		 = mref #[pInv.mref mref] #[mrefTV mref];
		# else
			 # #pInv.regularizer[2] = tikhregfun;
			 # #pInv.alpha[2]       = 1e-5*(10^k);
			 # #pInv.mref[:,2] 		= mref;
			 # pInv.regularizer = tikhregfun;
			 # pInv.alpha       = 1e-5*(10^k);
			 # pInv.mref 		= mref;
		# end
		# mc,Dc,flag,his = projGN(mc,pInv,pMisRFs,solveGN=projGNexplicit); #solveGN=projGNexplicit
		# #mc,Dc,flag,his = projGNCG(mc,pInv,pMisRFs,indCredit = [],dumpResults = dump);
		# myDump(mc,0,k+1,pInv,0,methodName,nDips,noiseAnglesDeg,noiseTrans,invertVis,nShots);
	# end
end

