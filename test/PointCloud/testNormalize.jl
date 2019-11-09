####### GRADIENT TEST:
u = rand(90);
(u_norm,J) = normalize(u);
display(u_norm)

u_0 = rand(90);
v = rand(90);

f_0,J = normalize(u_0);
for k=1:5
	h = 0.5^k;
	t, = normalize(u_0 .+ h.*v );
	tt = f_0 .+ h.*(J*v);
	println("E0 = ",norm(t - f_0),", E1 = ",norm(t - tt));
end



print(0);