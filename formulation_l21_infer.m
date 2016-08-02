function [acc] = formulation_l21_infer(W, W_x, data_val, label_val, opts)
    cte = opts.cte;
    yte = label_val;
    NumTstClass = length(unique(cte));
    C = NumTstClass;
    Xte = data_val;
    N = length(yte);
    y = zeros(N, 1);
    for n =1:N
        y(n) = find(cte==yte(n));
    end
   
    d = size(Xte, 2);
    matrix_path = opts.matrix_path;
    load(matrix_path);
    Z = PredicateMatrix(cte, :)';
    d_hat = size(Z, 1);

    X = Xte;
    A_ = (W_x * X')';
    pred_score = A_ * W * Z;
    [maxVal, maxIdx] = max(pred_score');
    pred_id = maxIdx';   
    GT_id = y;

    acc = sum(pred_id == GT_id) / length(y);


    fprintf('Accuracy = %1.4f%% (%d/%d)\n',100*acc,sum(pred_id == GT_id),numel(pred_id))


end
