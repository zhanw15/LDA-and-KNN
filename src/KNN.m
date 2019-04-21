% KNN k近邻算法 matlab实现
% add at 21-Apr-2019 by zhanw15
% ============= how to use it ======================
%% Data is the test Matrix. We get it target by knn.
%% 
%% Input && Target is Matrix we know it's target.
%% k is the most nearst k neighbour.
%%
%% Result = KNN( Data, Input, Target, k);
% ==================================================

function Result = KNN( Data, Input, Target, k)

	% Determine size of input anddata
	[n,~] = size(Input);
	[m,~] = size( Data);
   
    % Initialize, nSum is the total right target
    Result = 0;
	
    % Loop over every line to computing accuracy
    for i = 1:m
        
		% computing distance between every line
		Dis = repmat(Data(i,:),[n,1]) - Input;

		% sort to find the min k distance
		[~,I] = sort(sum(Dis.^2,2));
		
		% find the most show time of target
		Result = [Result; mode(Target(I(1:k)))];
    end
	
end

% EOF
