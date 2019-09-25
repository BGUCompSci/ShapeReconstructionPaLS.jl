export splitTwoRotationsTranslations
##
##This function assumes that mAll is [m,rotation_translation_1,rotation_translation_2]
##
function splitTwoRotationsTranslations(mAll::Array{Float64,1},numMoves1::Int64,numMoves2::Int64,choose::Int64)
numElementsMove1 = numMoves1*5;
numElementsMove2 = numMoves2*5;
m_len = length(mAll) - numElementsMove1 - numElementsMove2;

Ind = 0;

J = speye(Float32,length(mAll));
if choose==1
	Ind = 1:(m_len + numElementsMove1)
else
	Ind = [1:m_len;(m_len + numElementsMove1+1):length(mAll)];
end
return mAll[Ind],J[Ind,:];
end
