% LDA - MATLAB subroutine to perform linear discriminant analysis
% by Will Dwinnell and Deniz Sevis
% modified at 29-Oct-2018 by zhanw15

function W = LDA(Input,Target)

    % Determine size of input data
    [n,m] = size(Input);

    % Discover and count unique class labels 统计不同类别
    ClassLabel = unique(Target);
    k = length(ClassLabel);
    
    % Initialize
    nGroup     = NaN(k,1);           % Group counts        各类容量
    GroupMean  = NaN(k,m);           % Group sample means  各属性均值
    Sw         = zeros(m,m);         % Pooled covariance
    Sb         = zeros(m,m);         % Sb
    u          = zeros(1,m);         % All sample means

    % Loop over classes to perform intermediate calculations 计算Sw
    for i = 1:k
        % Establish location and size of each class 标识当前类 && 计算类容量
        Group      = (Target == ClassLabel(i));
        nGroup(i)  = sum(double(Group));
        
        % Calculate group mean vectors 计算各类各属性均值
        GroupMean(i,:) = mean(Input(Group,:));

        % Accumulate pooled covariance information 计算Sw
        Sw = Sw + (nGroup(i) - 1).* cov(Input(Group,:));
 
        u = u + nGroup(i)*GroupMean(i,:)/n; % 计算均值
    end

    for i = 1:k     % 计算 Sb
        Sb = Sb + nGroup(i)*( GroupMean(i,:)-u)'*( GroupMean(i,:)-u);
    end
    
    % 对特征向量进行排序
    [t,a] =        eig( Sb, Sw);
    b     = [ abs(diag(a)), t'];

    % 按照特征值大小对特征向量进行排序，并选择特征向量
    b = sortrows(     b,   'descend')';
    W =        b( 2:m+1,       1:k-1) ;

end

% EOF
