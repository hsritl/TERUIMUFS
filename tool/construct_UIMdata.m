function [result,missing_num,X_missing,zero_indices,one_indices] = construct_UIMdata(data,per)
% Each view of *data* is a matrix of size n*dv. 
% *per* refers to the specified average missing percentage.
view_num = length(data);
% The missing rate of each view under unbalanced conditions (corresponding to the article).
ratio{1}=1;
ratio{2}=[0.5,1.5];
ratio{3}=[0.5,1,1.5];
ratio{4}=[0.25,0.75,1.25,1.75];
ratio{5}=[0.25,0.75,1,1.25,1.75];
ratio{6}=[0.25,0.5,1,1,1.5,1.75];
instance_num=size(data{1},1);
perDel=0.01*per*ratio{view_num};
for view_idx=1:view_num
    missing_num(view_idx)=fix(instance_num*perDel(view_idx)); % *missing_num* stores the number of missing samples in each view.
    oz_indices{view_idx}=ones(1,instance_num);
    zero_indices{view_idx}= randperm(instance_num, missing_num(view_idx)); % The index of the missing samples in each view.
    oz_indices{view_idx}(zero_indices{view_idx}) = 0;
    one_indices{view_idx} = find(oz_indices{view_idx} == 1); % The index of the existing samples in each view.
    % result{view_idx} = NormalizeFea(data{view_idx}, 1);
    result{view_idx}=data{view_idx};
    X_missing{view_idx}=result{view_idx};
    result{view_idx}(zero_indices{view_idx},:) = 0;
    X_missing{view_idx}(zero_indices{view_idx},:) = [];
    result{view_idx}=result{view_idx}'; % Each view of *result* is a matrix of size dv*n, with missing samples filled with 0.
    X_missing{view_idx}=X_missing{view_idx}'; % Each view of *X_missing*is a matrix of size dv*nv, with the missing samples removed.
end
