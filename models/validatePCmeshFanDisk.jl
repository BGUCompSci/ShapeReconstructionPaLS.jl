
# The PC is centered on a rectangular grid, while the grid-based FD sits in a cubic grid.


using DelimitedFiles
using MAT
# using DSP

function smoothObj(Obj,times)
Obj = copy(Obj);
for k = 1:times
	Obj[2:end-1,2:end-1,2:end-1] = 	(Obj[2:end-1,2:end-1,2:end-1].*3.0 + 
									Obj[3:end  ,2:end-1,2:end-1] +
									Obj[1:end-2,2:end-1,2:end-1] +									
									Obj[2:end-1,1:end-2,2:end-1] + 
									Obj[2:end-1,3:end  ,2:end-1] + 									
									Obj[2:end-1,2:end-1,1:end-2] + 
									Obj[2:end-1,2:end-1,3:end  ])./9.0;
end
return Obj;
end

# function smoothObj9(Obj,times)
# for k=1:times
	# Obj = conv([1;2;1]/4.0,[1;2;1]/4.0,Obj);
# end
# return Obj;
# end



PC_orig = readdlm("fandisksmall.xyz");
Ns = PC_orig[:,4:6];
PC = PC_orig[:,1:3];
println(maximum(PC,dims=1));
# PC = PC*Diagonal(0.5./maximum(PC,dims=1)[:]);
println(maximum(PC,dims=1));
F = matopen("fandisksmall.mat");
B = read(F,"B");
n = size(B);
println(n)
pad = div.([400;400;400] .- collect(n),2) .+ 20;
B_pad = zeros(n[1]+2*pad[1],n[2]+2*pad[2],n[3]+2*pad[3]);
B_pad[pad[1]+1:end-pad[1],pad[2]+1:end-pad[2],pad[3]+1:end-pad[3]] .= B;
println("Smooth!")
B_pad = smoothObj(B_pad,3);

println(minimum(PC.+[23/50.0 12.75/50.0 25.0/50.0],dims=1))
println(maximum(PC.+[23/50.0 12.75/50.0 25.0/50.0],dims=1))

PC_subs = ceil.(Int64,(PC.+[23/50.0 12.75/50.0 25.0/50.0]).*400) .+ pad';
println("starting test");
for k = 1:size(PC_subs,1)
	testval = B_pad[PC_subs[k,1],PC_subs[k,2],PC_subs[k,3]];
	if testval >= 1.0 - 1e-8 || testval < 1e-8
		display(B_pad[PC_subs[k,1].+ (-1:1),PC_subs[k,2].+ (-1:1),PC_subs[k,3].+ (-1:1)]);
		println(testval)
		error("IM HERE");
	end
end




