export prepareSyntheticDipDataFiles
function prepareSyntheticDipDataFiles(m::Array{Float64},Minv::RegularMesh,filename::String,pad::Int64,theta_phi_rad::Array{Float64,2},samplingBinning::Int64, noiseSample::Float64, noiseAngles::Float64,noiseTrans::Float64,method = MATFree)

n = Minv.n;
pFor = getDipParam(Minv,theta_phi_rad + noiseAngles*randn(size(theta_phi_rad)),noiseTrans*randn(size(theta_phi_rad,1),3),samplingBinning,nworkers(),method);
@time Dremote,pFor = getData(m[:],pFor);

Dlocal = arrangeRemoteCallDataIntoLocalData(Dremote);
Data = Dlocal + noiseSample*mean(Dlocal[:])*randn(size(Dlocal));

pFor = getDipParam(Minv,theta_phi_rad + zeros(size(theta_phi_rad)),zeros(size(theta_phi_rad,1),3),samplingBinning,nworkers(),method);
Dremote,pFor = getData(m[:],pFor);

Dlocal = arrangeRemoteCallDataIntoLocalData(Dremote);
DataClean = Dlocal;





file = matopen(string(filename,"_data.mat"), "w");
write(file,"domain",Minv.domain);
write(file,"noiseSample",noiseSample);
write(file,"noiseAngles",noiseAngles);
write(file,"n",Minv.n);
write(file,"samplingBinning",samplingBinning);
write(file,"Data",Data);
write(file,"DataClean",DataClean);
write(file,"theta_phi_rad",theta_phi_rad);
# write(file,"m_true",m);
write(file,"pad",pad);
close(file);
return;
end

export readDipDataFile
function readDipDataFile(filename::String)
file = matopen(string(filename,"_data.mat"), "r");
domain = read(file,"domain");
n = read(file,"n");
samplingBinning = read(file,"samplingBinning");
Data = read(file,"Data");
# m_true = read(file,"m_true")
pad = read(file,"pad");
theta_phi_rad = read(file,"theta_phi_rad");
close(file);
return n,domain,samplingBinning,Data,theta_phi_rad,pad;
end