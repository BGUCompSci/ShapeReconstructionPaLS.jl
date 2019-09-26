export prepareSyntheticVisDataFiles

function prepareSyntheticVisDataFiles(m,Minv::RegularMesh,MScreen::RegularMesh,pad::Int64,theta_phi_rad::Array{Float64,2},
					XlocScreen::Float64,LocCamera::Array{Float64},filename::String,nShots::Int64, 
					noiseCamLoc::Float64, noiseScreenLoc::Float64, noiseAngles::Float64,noiseTrans::Float64,method::String)
n = Minv.n;
m = reshape(m,tuple(n...));
pFor = getSfSParam(Minv,theta_phi_rad + noiseAngles*randn(size(theta_phi_rad)),noiseTrans*randn(size(theta_phi_rad,1),3) ,MScreen,XlocScreen + noiseScreenLoc*randn(),
						LocCamera+noiseCamLoc*randn(size(LocCamera)),method,1);
Dremote,pFor = getData(m[:],pFor);
Data = arrangeRemoteCallDataIntoLocalData(Dremote);


# Data = Data + noiseSample*mean(Dlocal[:])*randn(size(Dlocal));				
					
file = matopen(string(filename,"_data.mat"), "w");
write(file,"domain",Minv.domain);
write(file,"n",Minv.n);
write(file,"noiseScreenLoc",noiseScreenLoc);
write(file,"noiseAngles",noiseAngles);
write(file,"noiseCamLoc",noiseCamLoc);
write(file,"ScreenDomain",MScreen.domain);
write(file,"n_screen",MScreen.n);
write(file,"CameraLoc",LocCamera);
write(file,"XScreenLoc",XlocScreen);
write(file,"Data",Data);
write(file,"theta_phi",theta_phi_rad);
# write(file,"m_true",m);
write(file,"pad",pad);
close(file);
return;
end


export readVisDataFile
function readVisDataFile(filename::String)
file = matopen(string(filename,"_data.mat"), "r");
domain = read(file,"domain");
n = read(file,"n");
Data = read(file,"Data");
# m_true = read(file,"m_true")
theta_phi = read(file,"theta_phi");
XScreenLoc = read(file,"XScreenLoc");
CameraLoc = read(file,"CameraLoc");
ScreenDomain = read(file,"ScreenDomain");
n_screen = read(file,"n_screen");
pad = read(file,"pad");
close(file);
return n,domain,XScreenLoc,CameraLoc,ScreenDomain,n_screen,Data,theta_phi,pad;
end