using LinearAlgebra
using Statistics
using DelimitedFiles;
using jInv.Mesh;
using MAT;
using jInv
using jInv.InverseSolve
using ShapeReconstructionPaLS
using ShapeReconstructionPaLS.PointCloud;
using ParamLevelSet;
using ShapeReconstructionPaLS.ShapeFromSilhouette;
 using ShapeReconstructionPaLS.Utils;
using Statistics
using Distributed
using SparseArrays
using DelimitedFiles
using Random

ENV["MPLBACKEND"] = "Qt4Agg"

plotting = true;

if plotting
    using PyPlot;
    using jInvVisPyPlot;
    close("all");
end

####
#Parameters to choose if: inversion with dip only/visual hull only/ joint inversion
invertPC = false;
invertVis = true;

#Place new added RBFs according to gradient values or randomly:
locateRBFwithGrads = invertPC;
pad_in_pixels = 25;
n = [100,100,100].+2*pad_in_pixels;
Mesh = getRegularMesh([0.0 1.5 0.0 1.5 0.0 1.5],n);

################################################
### Reading the model and pad ##################
################################################
model = "fandisksmall";
if invertPC
	file_name = string(pwd(),"/../models/",model,".xyz");
	PC_orig = readdlm(file_name);
	Normals = PC_orig[:,4:6];
	PC = PC_orig[:,1:3];
	#PC = PC*Diagonal(0.5./maximum(PC,dims=1)[:]);
	PC = PC .+ 0.5*(Mesh.domain[2] + Mesh.domain[1]);
	##Sort the PC:
	PC  = [ PC Normals];
	PC = PC[sortperm(PC[:, 1]), :];
	writedlm("SortedPC.txt",PC);
	println("PC:",size(PC)," ", PC[end,:]);
	if plotting
		scatter3D(Vector(PC[:,1]),Vector(PC[:,2]),Vector(PC[:,3]),s=0.1,alpha=1.0,color="blue")
	end
end


global m = 0;
if invertVis
	##Read mat file for for visual hull:
	file = matopen(string(pwd(),"/../models/",model,".mat"));
	A = read(file, "B");
	close(file);
	A = convert(Array{Float32,3},A[1:4:end,1:4:end,1:4:end]); # that's 100^3.
	m = zeros(Float64,tuple(n...))
	# m[pad_in_pixels+1:end-pad_in_pixels,pad_in_pixels+1:end-pad_in_pixels,pad_in_pixels+1:end-pad_in_pixels] = A;
	pad = div.(collect(n) .- collect(size(A)),2);
	m[pad[1]+1:end-pad[1],pad[2]+1:end-pad[2]-1,pad[3]+1:end-pad[3]] .= A;
	A=[];
end
# if plotting
    # plotModel(m);
    # savefig("TrueModel.png");
# end

################################################################
### Preparing the synthetic data using mat-free implementation
################################################################
ScreenMesh = getRegularMesh(Mesh.domain[3:6],n[2:3]); ## for vis.
ScreenDomain = ScreenMesh.domain;
n_screen = ScreenMesh.n;

#########################################################################################################
### Point Cloud Data ########################################################################################
#########################################################################################################

npc = 0;
nShots = 0;
noiseTrans = 0.0;
noiseAngles = 1.0;
if invertPC
	println("Data gen resolution:")
	println(n)
	dipDataFilename = string("pointCloudData",model,"_dataRes",n,1);
	trans = [0.0 0 0 ; 0.0 0 0 ]
	theta_phi_dip = [0.0 0.0 ; 0.0 0.0];
	noiseTrans = 0.00;
	noiseAngles = deg2rad(noiseAngles);
	npc = 2;
	prepareSyntheticPointCloudData(PC,Mesh,npc,theta_phi_dip,trans,noiseTrans,noiseAngles,dipDataFilename);
	
	#########################################################################################################
	### Read the dipping data ###############################################################################
	#########################################################################################################
	DataPC,P,Normals,trans,theta_phi_dip,npc = readPointCloudDataFile(dipDataFilename);
	### Set up the dip inversion
	P =  Array{Array{Float64,2}}(P);
	Normals = Array{Array{Float64,2}}(Normals);

end


####################################


#################################################################################################################
### Generate and Read the visual data ###############################################################################
#################################################################################################################
samplingBinning = 1;
if invertVis
	nShots = 2;
	noiseSample = 0.0;
	factorShotsCreation = 4;
	theta_phi_vis = deg2rad.(convert(Array{Float64,2}, [rand(0:359,factorShotsCreation*nShots)  rand(0:90,factorShotsCreation*nShots)]));
	#theta_phi_vis = deg2rad.(convert(Array{Float64,2},[[30.0 60.0] [45.0 90.0]]));
	VisDataFilename = string("visData_",model,"_dataRes",n,"_noiseSampleAnglesTrans",[noiseSample;noiseAngles;noiseTrans],"_shots",nShots);
	
	XlocScreen = Mesh.domain[2];
	LocCamera = [-12.0*(Mesh.domain[2] - Mesh.domain[1]) ; 0.0 ;0.0]; ## 0.0 here is the middle of the domain
	noiseCamLoc = 0.0;
	noiseScreenLoc = 0.0;
	prepareSyntheticVisDataFiles(m,Mesh,ScreenMesh,pad_in_pixels,theta_phi_vis,XlocScreen,LocCamera,
	VisDataFilename,factorShotsCreation*nShots, noiseCamLoc, noiseScreenLoc, noiseAngles,noiseTrans,MATFree);
	
	##################################################################################################################
	##################################################################################################################
	
	n,domain,XScreenLoc,CameraLoc,ScreenDomain,n_screen,DataVis,theta_phi_vis = readVisDataFile(VisDataFilename);
	norms = vec(sum(DataVis,dims = 1));
	idsorted = sortperm(norms);
	theta_phi_vis = theta_phi_vis[idsorted[(end-nShots+1):end],:];
	DataVis = DataVis[:,idsorted[(end-nShots+1):end]];

	ScreenMesh 	= getRegularMesh(ScreenDomain,div.(n_screen,samplingBinning));
	Mesh = getRegularMesh(domain,div.(n,samplingBinning));
	# pad = div(pad,samplingBinning);
	if  invertVis #samplingBinning==2 &&
		#DataVis = coarsenByTwo(DataVis,ScreenMesh.n,nShots);
		if plotting
			figure();
			for kkk = 1:nShots
				dk = DataVis[:,kkk];
				dk = reshape(dk,tuple(ScreenMesh.n...));
					subplot(8,8,kkk);
				plotModel(dk);
				
			end
			savefig(string("VisTrueModel.png"));
		end
	end
end

samplingBinning = 1;
println("Inv Mesh:")
println(Mesh.n)

method = RBF10Based; methodName="RBF10Based";

n_tup = tuple(Mesh.n...);
nWorkers = nworkers();
misfun = SSDFun; ## least squares


if invertPC
	### Set up the dip inversion
	# 0.0 below is margin : TODO: remove margin.
	pForPC = getPointCloudParam(Mesh,P,Normals,theta_phi_dip,trans,1,method);

	### Create Dip param if we use the GRAD of MATFREE to locate new RBFs
	if(locateRBFwithGrads)
		pForPC_MATFree = getPointCloudParam(Mesh,P,Normals,theta_phi_dip,trans,1,MATFree);
	end
	
	dobsPC = divideDataToWorkersData(1,DataPC);
	Wd_PC = Array{Array{Float32}}(undef,1);
	for k=1:1
		Wd_PC[k] = ones(Float32,size(dobsPC[k]));
	end
end

if invertVis
	### Set up the vis inversion
	pForVis = getSfSParam(Mesh,theta_phi_vis,zeros(nShots,3),ScreenMesh,XlocScreen,LocCamera,method,nWorkers);
	dobsVis = divideDataToWorkersData(nWorkers,DataVis);
	Wd_Vis = Array{Array{Float32}}(undef,nWorkers);
	for k=1:length(Wd_Vis)
		println("sqrt:",0.1*(Mesh.domain[2]-Mesh.domain[1])*prod(Mesh.h[2:3]));
		Wd_Vis[k] = 1e+2*sqrt((Mesh.domain[2]-Mesh.domain[1])*prod(Mesh.h[2:3]))*ones(Float32,size(dobsVis[k]));
		# Wd_Vis[k] = sqrt(0.3)*ones(Float32,size(dobsVis[k]));
		
	end
end

if invertVis && invertPC
	n_Moves_all = nShots + npc;
elseif invertPC
	n_Moves_all = npc;
else
	n_Moves_all = nShots;
end



#########################################################################################################
############### Volumetric inversion (pixel-wise) #######################################################
#########################################################################################################
if method == MATBased || method == MATFree
	error("Not relevant for PC.")
else
	nRBF = 10;
	
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
		modfunForPlotting = (m)->MeshFreeParamLevelSetModelFunc(Mesh,m;computeJacobian = 0,
		sigma = getDefaultHeaviside() ,bf = 1, numParamOfRBF = numParamOfRBF);
	else
		modfunForPlotting = (m)->MeshFreeParamLevelSetModelFunc(Mesh,m[1:(numParamOfRBF*nRBF)];computeJacobian = 0,
		sigma = getDefaultHeaviside() ,bf = 1, numParamOfRBF = numParamOfRBF);
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
	mref[1:numParamOfRBF:n_m_simple] .= ParamLevelSet.centerHeavySide*2.0;
	if isRBF10
		mref[2:10:n_m_simple] .= 15.0;
		mref[5:10:n_m_simple] .= 15.0;
		mref[7:10:n_m_simple] .= 15.0;
		boundsLow[2:10:n_m_simple] .= 0.05;
		boundsLow[5:10:n_m_simple] .= 0.05;
		boundsLow[7:10:n_m_simple] .= 0.05;
	else
		mref[2:5:n_m_simple] .= 2.0;
		boundsLow[2:5:n_m_simple] .= 0.05;
	end
	mref[(numParamOfRBF-2):numParamOfRBF:n_m_simple] .= (Mesh.domain[1] + Mesh.domain[2])/2;
	mref[(numParamOfRBF-1):numParamOfRBF:n_m_simple] .= (Mesh.domain[3] + Mesh.domain[4])/2;
	mref[numParamOfRBF:numParamOfRBF:n_m_simple] .= (Mesh.domain[5] + Mesh.domain[6])/2;
	m0[:] = mref[1:n_m_simple];
	if !isSimple
		if invertPC
			m0 = wrapRBFparamAndRotationsTranslations(m0,theta_phi_dip,trans);
		end
		if invertVis
			m0 = wrapRBFparamAndRotationsTranslations(m0,theta_phi_vis,zeros(size(theta_phi_vis,1),3));
		end
	end
	mref[:] = m0;
	mref[(n_m_simple+1):(n_m_simple+5*npc)].= 0.0;
	m0[(numParamOfRBF-2):numParamOfRBF:n_m_simple] .+= 0.1*(Mesh.domain[2] - Mesh.domain[1])*randn(nRBF);
	m0[(numParamOfRBF-1):numParamOfRBF:n_m_simple] .+= 0.1*(Mesh.domain[4] - Mesh.domain[3])*randn(nRBF);
	m0[numParamOfRBF:numParamOfRBF:n_m_simple]     .+= 0.1*(Mesh.domain[6] - Mesh.domain[5])*randn(nRBF);
	
	# alpha = 5e+0; #for fandisksmall
	alpha = 5e+1; #for fandiskpc
	alpha = 5e+2; # for vis? 
	
	II = (sparse(1.0I, nAll,nAll));
	II.nzval[(n_m_simple+1):(n_m_simple+5*npc)] .= 1e+3;
	regfun = (m, mref, M)->TikhonovReg(m,mref,M,II);
	# HesPrec = getEmptyRegularizationPreconditioner();
	HesPrec = getSSORCGRegularizationPreconditioner(1.0,1e-2,1);
	
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
	
	pMis_Free = getMisfitParam(pForPC_MATFree, Wd_PC, dobsPC, misfun, Iact_free,sback);
end
if invertVis && invertPC
	if isSimple
		Wd   = [Wd_PC;Wd_Vis];
		dobs = [dobsPC;dobsVis];
		pFor = [pForPC;pForVis];
		pMisRFs = getMisfitParam(pFor, Wd, dobs, misfun, Iact,sback);
	else
		modfun1 = (mAll) -> splitTwoRotationsTranslations(mAll,npc,nShots,1);
		modfun2 = (mAll) -> splitTwoRotationsTranslations(mAll,npc,nShots,2);
		pMisRFs1 = getMisfitParam(pForPC, Wd_PC, dobsPC, misfun, Iact,sback,ones(length(pForPC)),modfun1);
		pMisRFs2 = getMisfitParam(pForVis, Wd_Vis, dobsVis, misfun, Iact,sback,ones(length(pForVis)),modfun2);
		pMisRFs = [pMisRFs1;pMisRFs2];
		pMisRFs1 = 0;
		pMisRFs2 = 0;
	end
	n_Moves_all = nShots + npc;
	
elseif invertPC
	pMisRFs = getMisfitParam(pForPC, Wd_PC, dobsPC, misfun, Iact,sback);
else
	pMisRFs = getMisfitParam(pForVis, Wd_Vis, dobsVis, misfun, Iact,sback);
end

        
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
@everywhere function updateAngles(pFor::RemoteChannel,theta_phi)
pForMATFREE  = take!(pFor)
worker_theta_phi = theta_phi[pForMATFREE.workerSubIdxs,:]
pForMATFREE.theta_phi_rad=copy(worker_theta_phi);
put!(pFor,pForMATFREE)
return pFor;
end


function myDump(mc::Vector,Dc,iter,pInv,pMis,method="",npc=0,noiseAngle=0,noiseTranslation=0,useVisual=false,nshots=4,plotFun=modfunForPlotting)
	ntup = tuple((pInv.MInv.n)...);
	fullMc = plotFun(mc)[1];
	fullMc = reshape(IactPlot*fullMc[:],ntup);
	# fullMc = reshape(fullMc[:],ntup);
	close(888);
	figure(888);
	#plotModel(fullMc,true,pInv.MInv);
	plotModel(fullMc,includeMeshInfo = true,M_regular = pInv.MInv);
	sleep(1.0);
	savefig(string("It",iter,"_",ntup,".png"));
	writedlm(string("Params_iter_",iter,".dat"),convert(Array{Float16},mc));
	writedlm(string("Volume_iter_",iter,"_",ntup,".dat"),convert(Array{Float16},fullMc));
	

end


function plotVisData(Dc)
	close("all");
	close(999);
	figure(999);
	for kkk = 1:nShots
		dk = Dc[:,kkk];
		dk = reshape(dk,tuple(ScreenMesh.n...));
		subplot(2,2,kkk);
		plotModel(dk);
	end
end

cgit  = 50; # not used.
maxit = 5;
pcgTol = 1e-2; #not used

pInv = getInverseParam(Mesh,modfun,regfun,alpha,mref,boundsLow,boundsHigh,
maxStep = 1e-1,pcgMaxIter=cgit,pcgTol=pcgTol,
minUpdate = 0.01, maxIter = maxit,HesPrec=HesPrec);


#### Projected Gauss Newton
mc = m0;
Dc = nothing
pInv.maxIter = 10;
global mc,Dc,flag,his = projGN(mc,pInv,pMisRFs,solveGN=projGNexplicit);
pInv.maxIter = 10;
myDump(mc,0,1,pInv,0,methodName,npc,noiseAngles,noiseTrans,invertVis,nShots);






if !(method == MATBased || method == MATFree)
	outerIter = 80;
	grad_u = nothing;
	new_RBF_location = [];
	@sync for iterNum=2:outerIter
		new_nRBF = 10;
		println("Inversion using ", nRBF + new_nRBF," basis functions")
		# Dc - the data using our method.
		global Dc = Dc;
		
		#Lets compute the gradient of the misfit function:
		# locateRBFwithGrads = true;
		print(mc[end-5*n_Moves_all+1:end])
		if(locateRBFwithGrads)
			#(m11,theta_phi1,b1) 	= splitRBFparamAndRotationsTranslations(mc,nRBF,ndips,numParamOfRBF)
			#updateAngles(pForDip_MATFree,theta_phi1);
			pMis_Free = getMisfitParam(pForPC_MATFree, Wd_PC, dobsPC, misfun, Iact,sback);
			sigmaH = getDefaultHeaviside();
			allPoints = [P[1] ; P[2]];
			Dd, = computeMisfit(mc,fetch(pMisRFs[1])); #first element is pmis of PC		
			#gradMis = (fetch(Dd[1])-DataPC).^2;	
			gradMis = ((Dd)-DataPC).^2;
			sp = sortperm(gradMis,rev = true);
			sp = sp[1:round(Int64,length(sp)/4.0)];
			sp = sp[randperm(length(sp))];
			idxs = sp[1:new_nRBF];
			bwdSubs = [P[1] ; P[2]];
			println("all P size:",size(allPoints));
			Ii = allPoints[idxs,:];
		else
			u = modfunForPlotting(mc)[1];
			#Ii = findall(x -> x >0.2 && x<0.5,u);
			Ii = findall(x -> x >0.1 && x<0.9,u);
			if length(Ii) > new_nRBF
				Ii = Ii[randperm(length(Ii))[1:new_nRBF]];
			end
		end
    #while(Ii in new_RBF_location || Ii.-1 in new_RBF_location || Ii.+1 in new_RBF_location)
    #  Ii = findall(x -> x .>0.2 && x.<0.5,u);
    #  Ii = Ii[randperm(length(Ii))[1:new_nRBF]];
    #end
    	# println("length of I",size(Ii));
		global new_RBF_location = append!(new_RBF_location,Ii);
    
		if(locateRBFwithGrads)
			# println("Added rbfs at locations:",Ii);
			X1 = Ii[:,1]; X2 = Ii[:,2]; X3 = Ii[:,3];
		else
			println("ii:",Ii)
			println(ind2subv(n_tup,Ii))
			indices = getindex.(ind2subv(n_tup,Ii),[1 2 3 ]);
			println(indices)
			
			
			I1 = indices[:,1];
			I2 = indices[:,2];
			I3 = indices[:,3];
			
			X1 = I1*Mesh.h[1] .- 0.5*Mesh.h[1] .+ Mesh.domain[1];
			X2 = I2*Mesh.h[2] .- 0.5*Mesh.h[2] .+ Mesh.domain[3];
			X3 = I3*Mesh.h[3] .- 0.5*Mesh.h[3] .+ Mesh.domain[5];
		end
		# println("X1:",X1);println("X2:",X2);println("X3:",X3);
    
    
		n_mc_old = numParamOfRBF*nRBF;
		n_mc_new = numParamOfRBF*nRBF + size(Ii,1)*numParamOfRBF;
		mc_new = zeros(length(mc) + size(Ii,1)*numParamOfRBF);
		if isRBF10
			mref = pInv.mref[:,1];
		else
			mref = pInv.mref;
		end
		mref_new = zeros(Float32,length(mc) + size(Ii,1)*numParamOfRBF);

		mc_new[1:n_mc_old] = mc[1:n_mc_old];

		mc_new[(n_mc_new+1):end] = mc[(n_mc_old+1):end];
		mref_new[(n_mc_new+1):end] = mref[(n_mc_old+1):end];

		mc_new[(n_mc_old+1):numParamOfRBF:n_mc_new] .= 0.0; #ParamLevelSet.centerHeavySide - ParamLevelSet.deltaHeavySide;
		mc_new[(n_mc_old+numParamOfRBF-2):numParamOfRBF:n_mc_new] = X1;
		mc_new[(n_mc_old+numParamOfRBF-1):numParamOfRBF:n_mc_new] = X2;
		mc_new[(n_mc_old+numParamOfRBF):numParamOfRBF:n_mc_new] = X3;

		boundsLow = zeros(Float32,length(mc_new)).+a;
		boundsHigh = zeros(Float32,length(mc_new)).+b;
		boundsLow[1:n_mc_old] = pInv.boundsLow[1:n_mc_old];
		boundsHigh[1:n_mc_old] = pInv.boundsHigh[1:n_mc_old];
		
		eeps =  1.0/(0.01171875*2.5)
		if isRBF10
			# mc_new[(n_mc_old+2):10:n_mc_new] .= eeps #10.0;
			# mc_new[(n_mc_old+5):10:n_mc_new] .= eeps#10.0;
			# mc_new[(n_mc_old+7):10:n_mc_new] .= eeps#10.0;
			mc_new[(n_mc_old+2):10:n_mc_new] .= 10.0;
			mc_new[(n_mc_old+5):10:n_mc_new] .= 10.0;
			mc_new[(n_mc_old+7):10:n_mc_new] .= 10.0;
			
			boundsLow[(n_mc_old+2):10:n_mc_new] .= 0.05;
			boundsLow[(n_mc_old+5):10:n_mc_new] .= 0.05;
			boundsLow[(n_mc_old+7):10:n_mc_new] .= 0.05;
		else
			mc_new[(n_mc_old+2):5:n_mc_new] .= 6.0;
			boundsLow[(n_mc_old+2):5:n_mc_new] .= 0.05;
		end
		# boundsLow[end-9:end] .= -0.01;
		# boundsHigh[end-9:end] .= 0.01;
		pInv.boundsLow = boundsLow;
		pInv.boundsHigh = boundsHigh;
		mref_new[1:n_mc_new] = mc_new[1:n_mc_new];
		# mref_new[end-5*n_Moves_all+1:end-5*(n_Moves_all - nShots)].=0.0;
		mref_new[(n_mc_new+1):(n_mc_new+5*npc)].=0.0;
		global mc = mc_new;
		global nRBF += new_nRBF;


		#II = speye(Float32,length(mc_new));
		len = Int64(length(mc_new));
		IIs = sparse(1.0I,len,len);
		IIs.nzval[(n_mc_new+1):(n_mc_new+5*npc)] .= 1e+3;
		pInv.regularizer = (m, mref, M)->TikhonovReg(m,mref,M,IIs);
		
		if isRBF10
			spd_reg = (m,mref,M) -> RBF_SPD_regularization(m,mref,nRBF);
			pInv.regularizer  = [pInv.regularizer;spd_reg];
			pInv.mref 			  = [mref_new zeros(length(mref_new))];
			pInv.alpha .*= 0.8;
		else
			pInv.mref = mref_new;
			pInv.alpha *= 0.8;
		end
		
		mc,Dc,flag,his = projGN(mc,pInv,pMisRFs,solveGN=projGNexplicit);
		#mc[end-5*npc+1:end] .= 0.0;
		myDump(mc,0,iterNum,pInv,0,methodName,0,noiseAngles,noiseTrans,invertVis);
		if plotting
			Dc = [fetch(Dc[1]) fetch(Dc[2])];
			plotVisData(Dc);
		end
		if iterNum==outerIter
			pInv.alpha *= 0.1;
			pInv.maxIter = 200;
			mc, = projGN(mc,pInv,pMisRFs,solveGN=projGNexplicit);
			myDump(mc,0,iterNum+1,pInv,0,methodName,0,noiseAngles,noiseTrans,invertVis);
		end
    
	end
end
