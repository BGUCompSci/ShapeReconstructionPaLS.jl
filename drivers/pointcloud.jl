using jInv.Mesh;
using MAT;
using jInv
using jInv.InverseSolve
using ShapeReconstructionPaLS
using ParamLevelSet;
using ShapeReconstructionPaLS.Utils;
using ShapeReconstructionPaLS.PointCloud;
using DelimitedFiles
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

locateRBFwithGrads = false;
n = [128,128,128];
mesh = getRegularMesh([0.0 1.5 0.5 1.5 0.5 1.5],n);
n_tup = collect(n);
n_tup = tuple(n_tup...)

################################################
### Reading the model and pad ##################
################################################

 model = "fandiskpc";
file = open(string(pwd(),"/models/",model,".xyz"));

#file = open("pc_full.xyz")
A = readlines(file);
A = split.(A);
B =  Array{Array{Float64,1},1}(undef,length(A));
N = Array{Array{Float64,1},1}(undef,length(A));
for i = 1:1:size(A,1)
	 global B[i] = parse.(Float64,A[i])[1:3] .+ 0.75;
	 global N[i] = parse.(Float64,A[i])[4:6];
end
close(file);
A= Array{Float64,2}(transpose(hcat(B...)));
Normals = Array{Float64,2}(transpose(hcat(N...)));
println("first line of xyz:",A[1,:]);
println("first line of normals:",Normals[1,:]);
npc = size(A,1); #Number of points in point cloud
println("A size:",size(A)); println("Normals size:",size(Normals));
B = [];


#tmp = [tuple(vec(A[i, :])...) for i in 1:size(A, 1)];
#indices = CartesianIndex.(tmp);
#linIdx = LinearIndices(tuple(n...))
#indices = linIdx[indices];
#Create indices matrix P:
#P should be k \times n where k = |Point cloud| , n = #vertices on grid
P = A;
println("Size of P:",size(P));

writedlm(string("originalPC.dat"),convert(Array{Float16},A));


#Clear vars:
indices = [];
A = []; tmp = [];
linIdx = [];


noiseAnglesDeg = 0.0;
#original was mean(2*Mesh.h);
noiseTrans = 0.0;
noiseAngles = deg2rad(noiseAnglesDeg);
#########################################################################################################
### Dipping Data ########################################################################################
#########################################################################################################


method = RBF10Based; methodName="RBF10Based";


nWorkers = nworkers();
println("nworkers=",nWorkers)
misfun = PCFun; ## least squares

	dipDataFilename = string("pointCloudData",model,"_dataRes",n,1);
	b = [0.0 0 0] #; 0.0 0 0 ]
	theta_phi_dip = [0.0 0.0] #; 0.0 0.0];
	noiseTrans = 0.0;
	noiseAngles = deg2rad(0.0);
	npc = 1;
	prepareSyntheticPointCloudData(P,Normals,npc,theta_phi_dip,b,noiseTrans,noiseAngles,dipDataFilename);

	#########################################################################################################
	### Read the dipping data ###############################################################################
	#########################################################################################################
	Data,P,Normals,Margin,b,theta_phi_dip,npc = readPointCloudDataFile(dipDataFilename);
	### Set up the dip inversion
	pForDip = getPointCloudParam(P[1],Normals,Margin,theta_phi_dip,b,nWorkers,method);
	ndips = npc;
	if(locateRBFwithGrads)
		pForDip_MATFree = getPointCloudParam(P,Normals,Margin,theta_phi_dip,b,nWorkers,MATFree);
	end
	
	### Create Dip param if we use the GRAD of MATFREE to locate new RBFs
	dobsDirect = divideDataToWorkersData(nWorkers,Data);
	println("size of dobs",size(dobsDirect))
	#nnz = filter(x -> x != 0,dobsDirect[1]);
	#println("length of dobsdirect: ",length(dobsDirect[1]),"_",length(nnz));
	Wd_Dip = Array{Array{Float32}}(undef,nWorkers);
	for k=1:nWorkers
		Wd_Dip[k] = ones(Float32,size(dobsDirect[k]));
	end
	println("Wd dip:",size(Wd_Dip))



n_Moves_all = npc;
println("npc = ",npc);


isSimple = 0;
isRBF10 = 0;



###############################################################################################
############## USE RBF dictionary #############################################################
###############################################################################################


	nRBF = 2;
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
		modfunForPlotting = (m)->MeshFreeParamLevelSetModelFunc(mesh,m ,computeJacobian = 0,
			sigma = getDefaultHeaviside() ,bf = 1, numParamOfRBF = numParamOfRBF);
	else
		modfunForPlotting = (m)->MeshFreeParamLevelSetModelFunc(mesh,m[1:(numParamOfRBF*nRBF)],computeJacobian = 0,
			sigma = getDefaultHeaviside() ,bf = 1, numParamOfRBF = numParamOfRBF);
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
		mref[2:5:n_m_simple] .= 4.0;
		boundsLow[2:5:n_m_simple] .= 0.05;
	end
	mref[(numParamOfRBF-2):numParamOfRBF:n_m_simple] .= 0.75;
	mref[(numParamOfRBF-1):numParamOfRBF:n_m_simple] .= 0.75;
	mref[numParamOfRBF:numParamOfRBF:n_m_simple] .= 0.75;
	m0[:] .= mref[1:n_m_simple];
	
	
	mref[:] = m0;
	m0[(numParamOfRBF-2):numParamOfRBF:n_m_simple] .+= 0.1*randn(nRBF);
	m0[(numParamOfRBF-1):numParamOfRBF:n_m_simple] .+= 0.1*randn(nRBF);
	m0[numParamOfRBF:numParamOfRBF:n_m_simple]     .+= 0.1*randn(nRBF);
	
	

	alpha = 5e+1;

	II = (sparse(1.0I, n_m_simple,n_m_simple));
	regfun = (m, mref, M)->TikhonovReg(m,mref,M,II);
	HesPrec = getEmptyRegularizationPreconditioner();

	if isRBF10
		spd_reg = (m,mref,M) -> RBF_SPD_regularization(m,mref,nRBF);
		regfun 	 		= [regfun;spd_reg];
		alpha 		 	= [alpha;1e-1*alpha]; #was 1e-4 for cube\prism !!!!
		mref 			= [mref zeros(size(mref))];
	end

pMisRFs = getMisfitParam(pForDip, Wd_Dip, dobsDirect, misfun, Iact,sback);




cgit  = 50; # not used.
maxit = 10;
pcgTol = 1e-2; #not used




function myDump(mc::Vector,Dc,iter,pInv,pMis,method="",ndips=0,noiseAngle=0,noiseTranslation=0,useVisual=false,plotFun=modfunForPlotting)
	# if resultsFilename!=""
		# Temp = splitext(resultsFilename);
		# Temp = string(Temp[1],"_GN",iter,Temp[2]);

			#ntup = tuple((pInv.MInv.n)...);
			ntup = tuple(n...)
			fullMc = plotFun(mc)[1];
			println("size of fullmc:",size(fullMc))
				writedlm(string("FandiskDirectParams_iter_",iter,".dat"),convert(Array{Float64},mc));
				writedlm(string("FandiskDirectVolume_iter_",iter,"_",ntup,".dat"),convert(Array{Float16},fullMc));
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
pInv = getInverseParam(mesh,modfun,regfun,alpha,mref,boundsLow,boundsHigh,
                     maxStep = 1e-1,pcgMaxIter=cgit,pcgTol=pcgTol,
					 minUpdate = 0.01, maxIter = maxit,HesPrec=HesPrec);

# end

#### Projected Gauss Newton
mc = m0;
pInv.maxIter = 10;
Dc = nothing;
mc,Dc,flag,his = projGN(mc,pInv,pMisRFs,solveGN=projGNexplicit);

print("DC shape:",size(Dc))

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
		new_nRBF = 10;
		if (times > 100)
		  #times=0;
		  new_nRBF = 0;
		end
		global times = times + 1;

		# Dc - the data using our method.
		global Dc = Dc;
		u = modfunForPlotting(mc)[1];
		#I = find((u.>0.3) .& (u.<0.6));
		#Ii = findall(x -> x >0.2 && x<0.5,u);
		#Lets compute the gradient of the misfit function:
		locateRBFwithGrads = true;
		if(locateRBFwithGrads)
			#(m11,theta_phi1,b1) 	= splitRBFparamAndRotationsTranslations(mc,nRBF,ndips,numParamOfRBF)
			#updateAngles(pForDip_MATFree,theta_phi1);
			pMis_Free = getMisfitParam(pForDip, Wd_Dip, dobsDirect, misfun, Iact,sback);
			#computeMisfit(mc,pMisRFs);
			#gradMis = computeGradMisfit(mc,Dc,pMisRFs)
			#gradMis = gradMis.^2;
			# println("gradMis size:",size(gradMis))
			# println("size gradmis:",size(gradMis))
			# amax = argmax(gradMis); vmax = maximum(gradMis);
			# sp = sortperm(gradMis,rev = true);
			# Ii = sp[1:new_nRBF];
			bwdSubs = P[1];
			println("all P size:",size(P[1]))
			bwdSubs = bwdSubs[round(Int64,0*size(bwdSubs,1)/2)+1:round(Int64,1*size(bwdSubs,1)/2),:];
			
			
			Ii = bwdSubs;
			Ii = Ii[randperm(size(Ii,1))[1:new_nRBF],:];
			println("Ii size after randperm:",size(Ii));
		end
		
		
		if(locateRBFwithGrads == false)
		if length(Ii) > new_nRBF
		Ii = Ii[randperm(length(Ii))[1:new_nRBF]];
		end
		end
    
    #while(Ii in new_RBF_location || Ii.-1 in new_RBF_location || Ii.+1 in new_RBF_location)
    #  Ii = findall(x -> x .>0.2 && x.<0.5,u);
    #  Ii = Ii[randperm(length(Ii))[1:new_nRBF]];
    #end
    		println("length of I",size(Ii));
		 global new_RBF_location = append!(new_RBF_location,Ii);
    
		if(locateRBFwithGrads)
			println("Added rbfs at locations:",Ii);
			X1 = Ii[:,1]; X2 = Ii[:,2]; X3 = Ii[:,3];
		else
		println("ii:",Ii)
		println(ind2subv(n_tup,Ii))
		indices = getindex.(ind2subv(n_tup,Ii),[1 2 3 ]);
		println(indices)
		
		
		I1 = indices[:,1];
		I2 = indices[:,2];
		I3 = indices[:,3];
		
		X1 = I1*mesh.h[1] .- 0.5*mesh.h[1] .+ mesh.domain[1];
		X2 = I2*mesh.h[2] .- 0.5*mesh.h[2] .+ mesh.domain[3];
		X3 = I3*mesh.h[3] .- 0.5*mesh.h[3] .+ mesh.domain[5];
		end
		println("X1:",X1);println("X2:",X2);println("X3:",X3);
    
    
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

		pInv.boundsHigh = zeros(Float32,length(mc_new)) .+ b;
		boundsLow = zeros(Float32,length(mc_new)) .+ a;
		boundsLow[1:n_mc_old] = pInv.boundsLow[1:n_mc_old];
		if isRBF10
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
		pInv.boundsLow = boundsLow;
		mref_new[1:n_mc_new] = mc_new[1:n_mc_new];

		global mc = mc_new;
		global nRBF += new_nRBF;


		#II = speye(Float32,length(mc_new));
		len = Int64(length(mc_new));
		println(typeof(len))
		IIs = sparse(1.0I,len,len);
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
		myDump(mc,0,iterNum,pInv,0,methodName,0,noiseAnglesDeg,noiseTrans,invertVis);
		
		if iterNum==outerIter
			pInv.alpha *= 0.1;
			pInv.maxIter = 200;
			mc, = projGN(mc,pInv,pMisRFs,solveGN=projGNexplicit);
			myDump(mc,0,iterNum+1,pInv,0,methodName,0,noiseAnglesDeg,noiseTrans,invertVis);
		end
    
	end
	
		meshForPlotting = getRegularMesh(Mesh.domain,Mesh.n*1);
		ntup = tuple((meshForPlotting.n)...);
		if isSimple
			modfunForPlotting = (m)->MeshFreeParamLevelSetModelFunc(meshForPlotting,m,Xc = P,computeJacobian = 0,
				sigma = getDefaultHeavySide() ,bf = 1, numParamOfRBF = numParamOfRBF);
		else
			modfunForPlotting = (m)->MeshFreeParamLevelSetModelFunc(meshForPlotting,m[1:(numParamOfRBF*nRBF)],Xc = P,computeJacobian = 0,
				sigma = getDefaultHeavySide() ,bf = 1, numParamOfRBF = numParamOfRBF);
		end
		fullMc = modfunForPlotting(mc)[1];
		writedlm(string("Fandisk200Direct_Volume_final_",ntup,"_from_",tuple((Mesh.n)...),".dat"),convert(Array{Float16},fullMc));
		#m = convert(Array{Float16},m);
		#writedlm(string("Fandisk200Direct_Volume_orig.dat"),m);
		#writedlm(string("Fandisk200Direct_RBF_centers.dat"),mCenters);
	
end

