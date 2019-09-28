using jInv.Mesh;
using MAT;
using jInv
using jInv.InverseSolve
using ShapeReconstructionPaLS
using ParamLevelSet;
using ShapeReconstructionPaLS.Utils;
using ShapeReconstructionPaLS.Direct;
using LinearAlgebra
using Statistics
using Distributed
using SparseArrays
using Random
ENV["MPLBACKEND"] = "Qt4Agg"

function ind2subv(shape, indices)
    CI = CartesianIndices(shape)
    return getindex.(Ref(CI), indices)
end

plotting = false;

if plotting
	using PyPlot;
	using jInvVis;
	close("all");
end


invertDip = true;
invertVis = false;

locateRBFwithGrads = true;


################################################
### Reading the model and pad ##################
################################################
#model = "skull_160"; ## skull
 model = "bone_160";
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
Mesh = getRegularMesh([0.0 10.0 0.0 10.0 0.0 10.0],n);


noiseAnglesDeg = 0.0;
#original was mean(2*Mesh.h);
noiseTrans = mean(0.0*Mesh.h);
noiseAngles = deg2rad(noiseAnglesDeg);
#########################################################################################################
### Dipping Data ########################################################################################
#########################################################################################################
samplingBinning = 1;


Mesh = getRegularMesh([0.0 10.0 0.0 10.0 0.0 10.0],div.(n,samplingBinning));
pad = div(pad,samplingBinning);

method = RBF10Based; methodName="RBF10Based";

samplingBinning = 1;
println("Inv mesh:")
println(Mesh.n)

n_tup = tuple(Mesh.n...);
nWorkers = nworkers();
println("nworkers=",nWorkers)
misfun = SSDFun; ## least squares


	### Set up the dip inversion
	pForDip = getDirectParam(Mesh,samplingBinning,nWorkers,method);

	### Create Dip param if we use the GRAD of MATFREE to locate new RBFs
	dobsDirect = divideDirectDataToWorkersData(nWorkers,m);
	println(size(dobsDirect))
	#nnz = filter(x -> x != 0,dobsDirect[1]);
	#println("length of dobsdirect: ",length(dobsDirect[1]),"_",length(nnz));
	Wd_Dip = Array{Array{Float32}}(undef,nWorkers);
	for k=1:nWorkers
		Wd_Dip[k] = ones(Float32,size(dobsDirect[k]));
	end
	println("Wd dip:",size(Wd_Dip))



# if invertVis
# ### Set up the vis inversion
	# pForVis = getVisHullParam(Mesh,theta_phi_vis,zeros(nShots,3),ScreenMesh,XlocScreen,LocCamera,method,nWorkers);
	# dobsVis = divideDataToWorkersData(nWorkers,DataVis);
	# Wd_Vis = Array{Array{Float32}}(nWorkers);
	# for k=1:length(Wd_Vis)
		# Wd_Vis[k] = sqrt(0.1*(Mesh.domain[2]-Mesh.domain[1])*prod(Mesh.h[2:3]))*ones(Float32,size(dobsVis[k]));
	# end
# end


n_Moves_all = 1;


isSimple = 0;
isRBF10 = 0;


###############################################################################################
############## USE RBF dictionary #############################################################
###############################################################################################


	nRBF = 20;

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
	boundsHigh = zeros(Float32,n_m_simple) .+ b;
	boundsLow = zeros(Float32,n_m_simple) .+ a;
	# m0 - initial guess. mref - reference for regularization.
	mref = zeros(Float32,n_m_simple);
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
	m0[:] .= mref[1:n_m_simple];
	
	
	mref[:] = m0;
	m0[(numParamOfRBF-2):numParamOfRBF:n_m_simple] .+= 0.1*(Mesh.domain[2] - Mesh.domain[1])*randn(nRBF);
	m0[(numParamOfRBF-1):numParamOfRBF:n_m_simple] .+= 0.1*(Mesh.domain[4] - Mesh.domain[3])*randn(nRBF);
	m0[numParamOfRBF:numParamOfRBF:n_m_simple]     .+= 0.1*(Mesh.domain[6] - Mesh.domain[5])*randn(nRBF);

	alpha = 5e-1;

	#II = speye(Float32,n_m_simple);
	II = (sparse(1.0I, n_m_simple,n_m_simple));
	println("size of II:",size(II));
	println("size of m:",size(m))
	regfun = (m, mref, M)->TikhonovReg(m,mref,M,II);
	HesPrec = getEmptyRegularizationPreconditioner();

	if isRBF10
		spd_reg = (m,mref,M) -> RBF_SPD_regularization(m,mref,nRBF);
		regfun 	 		= [regfun;spd_reg];
		alpha 		 	= [alpha;1e-1*alpha]; #was 1e-4 for cube\prism !!!!
		mref 			= [mref zeros(size(mref))];
	end

pMisRFs = getMisfitParam(pForDip, Wd_Dip, dobsDirect, misfun, Iact,sback);
# if(locateRBFwithGrads)



cgit  = 50; # not used.
maxit = 10;
pcgTol = 1e-2; #not used




function myDump(mc::Vector,Dc,iter,pInv,pMis,method="",ndips=0,noiseAngle=0,noiseTranslation=0,useVisual=false,plotFun=modfunForPlotting)
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
			
				savefig(string("FandiskDirectIt",iter,"_",ntup,".png"));
				writedlm(string("FandiskDirectParams_iter_",iter,".dat"),convert(Array{Float64},mc));
				writedlm(string("FandiskDirectVolume_iter_",iter,"_",ntup,".dat"),convert(Array{Float16},fullMc));

		end
	# end
end

pInv = getInverseParam(Mesh,modfun,regfun,alpha,mref,boundsLow,boundsHigh,
                     maxStep = 2e-1,pcgMaxIter=cgit,pcgTol=pcgTol,
					 minUpdate = 0.01, maxIter = maxit,HesPrec=HesPrec);

# end

#### Projected Gauss Newton
mc = m0;

  pInv.maxIter = 300;
	mc,Dc,flag,his = projGN(mc,pInv,pMisRFs,solveGN=projGNexplicit);

pInv.maxIter = 200;
myDump(mc,0,1,pInv,0,"",0,noiseAnglesDeg,noiseTrans,invertVis);



nRBF=nRBF;
times=0;


if !(method == MATBased || method == MATFree)
	outerIter = 40;
	grad_u = nothing;
	new_RBF_location = [];
	@sync for iterNum=2:outerIter
		println("Inversion using ", nRBF," basis functions")
		new_nRBF = 2;
		if (times > 28)
		  #times=0;
		  new_nRBF = 0;
		end
		global times = times + 1;

		# Dc - the data using our method.

		u = modfunForPlotting(mc)[1];
		#I = find((u.>0.3) .& (u.<0.6));
		Ii = findall(x -> x >0.3 && x<0.6,u);
		#Lets compute the gradient of the misfit function:
	  
   
		if length(Ii) > new_nRBF
		Ii = Ii[randperm(length(Ii))[1:new_nRBF]];
		end
    
    while(Ii in new_RBF_location || Ii.-1 in new_RBF_location || Ii.+1 in new_RBF_location)
      Ii = findall(x -> x .>0.3 && x.<0.6,u);
      Ii = Ii[randperm(length(Ii))[1:new_nRBF]];
    end
    		println("length of I",size(Ii));
		 global new_RBF_location = append!(new_RBF_location,Ii);
    
   
   
		#(I1,I2,I3) = ind2sub(n_tup,I);
		indices = getindex.(ind2subv(n_tup,Ii),[1 2]);
		println(indices)
		(I1,I2,I3) = indices;
		X1 = I1*Mesh.h[1] - 0.5*Mesh.h[1] + Mesh.domain[1];
		X2 = I2*Mesh.h[2] - 0.5*Mesh.h[2] + Mesh.domain[3];
		X3 = I3*Mesh.h[3] - 0.5*Mesh.h[3] + Mesh.domain[5];
    
    
    
    
		n_mc_old = numParamOfRBF*nRBF;
		n_mc_new = numParamOfRBF*nRBF + length(Ii)*numParamOfRBF;
		mc_new = zeros(length(mc) + length(Ii)*numParamOfRBF);
		if isRBF10
			mref = pInv.mref[:,1];
		end
		mref_new = zeros(Float32,length(mc) + length(Ii)*numParamOfRBF);

		mc_new[1:n_mc_old] = mc[1:n_mc_old];

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


		#II = speye(Float32,length(mc_new));
		len = Int64(length(mc_new));
		println(typeof(len))
		IIs = sparse(1.0I,len,len);
		pInv.regularizer = (m, mref, M)->TikhonovReg(m,mref,M,IIs);
		pInv.alpha .*= 0.8;
		if isRBF10
			spd_reg = (m,mref,M) -> RBF_SPD_regularization(m,mref,nRBF);
			pInv.regularizer  = [pInv.regularizer;spd_reg];
			pInv.mref 			  = [mref_new zeros(length(mref_new))];
		else
			pInv.mref = mref_new;
		end
		
		mc,Dc,flag,his = projGN(mc,pInv,pMisRFs,solveGN=projGNexplicit);
		myDump(mc,0,iterNum,pInv,0,methodName,0,noiseAnglesDeg,noiseTrans,invertVis);
		
		if iterNum==outerIter
			pInv.alpha *= 0.1;
			pInv.maxIter = 200;
			mc, = projGN(mc,pInv,pMisRFs,solveGN=projGNexplicit);
			myDump(mc,0,iterNum+1,pInv,0,methodName,0,noiseAnglesDeg,noiseTrans,invertVis);
		end
		global nRBF += new_nRBF;
    
	end
	
	## here I produce the shape on a twice finer mesh for Andrei to produce nice pictures.
	
		meshForPlotting = getRegularMesh(Mesh.domain,Mesh.n*8);
		ntup = tuple((meshForPlotting.n)...);
		if isSimple
			modfunForPlotting = (m)->ParamLevelSetModelFunc(meshForPlotting,m;computeJacobian = 0,
				sigma = getDefaultHeavySide() ,bf = 1, numParamOfRBF = numParamOfRBF);
		else
			modfunForPlotting = (m)->ParamLevelSetModelFunc(meshForPlotting,m[1:(numParamOfRBF*nRBF)];computeJacobian = 0,
				sigma = getDefaultHeavySide() ,bf = 1, numParamOfRBF = numParamOfRBF);
		end
		fullMc = modfunForPlotting(mc)[1];
		writedlm(string("Fandisk200Direct_Volume_final_",ntup,"_from_",tuple((Mesh.n)...),".dat"),convert(Array{Float16},fullMc));
		#m = convert(Array{Float16},m);
		#writedlm(string("Fandisk200Direct_Volume_orig.dat"),m);
		#writedlm(string("Fandisk200Direct_RBF_centers.dat"),mCenters);
	
end

