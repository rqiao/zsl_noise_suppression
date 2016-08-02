function formulation_l21(lx, lw, dataset_name)
    cvsplit = 0;
    if dataset_name =='AwA'
        datapath = [ './dataset/AwA'];
        matrix_path = [datapath, '/predicate-matrix-binary-1855-all.mat']; %path of candidate words
        if cvsplit==0
            % get original split
            tmp = load([datapath,'/constants.mat'],'trainclasses_id','testclasses_id');
            cte = tmp.testclasses_id';
            ctr = tmp.trainclasses_id';
            clear tmp
        else
            % build training-testing split
            cte = (cvsplit-1)*10+(1:10); % test classes
            ctr = setdiff(1:50,cte);     % training classes
        end
        load([datapath,'/constants.mat'])           
        load([datapath, '/AWA_vgg19_pool.mat']);
    elseif dataset_name =='CUB'
            datapath = ['./dataset/CUB_200_2011'];
            matrix_path = [datapath, '/predicate-matrix-binary-3905-all.mat'];
            load([datapath, '/image_class_labels.mat']);
            label = imageClassLabels(:, 2);
            load([datapath, '/train_test_split.mat']);
            ctr = train_cid;
            cte = test_cid;
            load([datapath, '/cnn_feat-imagenet-vgg-verydeep-19.mat']);
            Data = double(cnn_feat');
    end

    lambdaX = 10^lx;
    lambdaW = 10^lw;
    opts.lambdaX = lambdaX;
    opts.lambdaW = lambdaW;

    opts.matrix_path = matrix_path;

    opts.cte = cte;
    opts.ctr = ctr;
    NumTrnClass = length(unique(ctr));
    NumTstClass = length(unique(cte));
    fprintf('Load training set\n') 

    NumClass = NumTrnClass + NumTstClass;
    nPerClass = zeros(NumClass, 1);
    IdPerClass = cell(NumClass, 1);
    for idc = 1:NumClass

        IdPerClass{idc} = find(label==idc);
        nPerClass(idc) = sum(label==idc);

    end
    opts.nPerClass = nPerClass;
    Xtr = []; ytr = []; 
    for idc = ctr 

        Xc = Data(IdPerClass{idc}, :);
        Xtr = [Xtr; Xc];
        ytr = [ytr; idc*ones(size(Xc,1),1)];

    end
    Xte = []; yte = []; 
    for idc = cte 

        Xc = Data(IdPerClass{idc}, :);
        Xte = [Xte; Xc];
        yte = [yte; idc*ones(size(Xc,1),1)];

    end

    emmbeding_size = NumTrnClass;
    C = NumTrnClass;
    X = Xtr;
    N = length(ytr);
    Y = zeros(N, C);
    y = zeros(N, 1);
    for n =1:N
        Y(n, :) = ctr==ytr(n);
        y(n) = find(ctr==ytr(n));
    end

    d = size(X, 2);
    load(matrix_path);
    Z = PredicateMatrix(ctr, :)';

    d_hat = size(Z, 1);
    m = emmbeding_size;
    W_x = randn([m, d]);
    W = randn([d_hat, m]); 

    A_ = (W_x * X')';

    obj0 = objective_function1(A_, W, Z, Y, opts);
    D = eye(d_hat);
    max_iter = 5;
    max_iter_W = 5;
    Wdiff = zeros(max_iter_W, 1);
    fprintf('multiplying X\n');

    XX = full(X' * X);
    fprintf('inversing XX\n');
    XX_inv_reg = inv(XX + opts.lambdaX * eye(d));
    fprintf('inversing complete.\n');

    terminate_thresh = 0.001;
    objAll0 = obj0;

    for ITER = 1: max_iter
        O = W' * Z;
        W_x = iter_W_x(XX_inv_reg, X, O, Y, W, opts);
        A_ = (W_x * X')';

        objAll = objective_function_All(A_, W_x, W, Z, Y, opts);
        fprintf('obj All at ITER %d = %f \n', ITER,  objAll);
        obj0 = objAll;
        for iter = 1:max_iter_W

            W0 = W;
            W = iter_W(D, A_, Z, Y, W_x, opts);
            
            D = diag([1 ./ (2*normL2_by_row(W))]);

            Wdiff(iter) = norm(W - W0,'fro');
            obj1 = objective_function_All(A_, W_x, W, Z, Y, opts);
            fprintf('obj W at iter %d = %f \n', iter,  obj1);
            if abs(obj1-obj0) < terminate_thresh
                break;
            end
            obj0 = obj1;
        end

        fprintf('\n');

        fprintf('training error at ITER %d = %.6f \n', ITER,  get_error(A_, W', Z, y));

        [acc] = formulation_l21_infer(W', W_x, Xte, yte, opts);
        result_file = [dataset_name, '_results.txt'];
        if ~exist('./results/', 'dir')
            mkdir('./results/');
        end
        fid = fopen(['./results/',  result_file], 'a+');

        fprintf(fid, 'iter = %d, lx = %d, lw = %d, acc = %.4f \n',ITER, lx, lw, acc);
        fclose(fid);
        if abs(objAll-objAll0) < terminate_thresh
                break;
        end
        objAll0 = objAll;
    end
end


function value = objective_function1(A_, W, Z, Y, opts)

    value = trace((A_ * W' * Z - Y)' * (A_ * W' * Z - Y)) + opts.lambdaW * normL21(W);
end

function value = objective_function_All(A_, W_x, W, Z, Y, opts)
    value = trace((A_ * W' * Z - Y)' * (A_ * W' * Z - Y)) + opts.lambdaW * normL21(W) ...
     + opts.lambdaX * norm(W_x' * W' * Z, 'fro')^2;
end


function value = normL21(M)
    ep = 0.0001;
    value = sum(sqrt(sum(abs(M).^2,2)+ep));
end

function W = iter_W(D, A_, Z, Y, W_x, opts)

    AA = A_' * A_ ;
    A =  inv(AA + opts.lambdaX * W_x * W_x') * (opts.lambdaW  );
    B =  Z * Z' * inv(D);
    C =  inv(AA + opts.lambdaX * W_x * W_x') * A_' * Y * Z'* inv(D) ;

    W = sylvester(A,B, C)';
end

function W_x = iter_W_x(XX_inv_reg, X, O, Y, W, opts)
    W_x = (XX_inv_reg * full(X)' * Y * O' * inv(O*O'))';

end

function value = normL2_by_row(M)
    ep = 0.0001;
    value = sqrt(sum(M.^2,2) + ep);
end

function err = get_error(A_, W, Z, y)
    pred_score = A_ * W * Z;
    [maxVal, maxIdx] = max(pred_score');
    pred_id = maxIdx';
    GT_id = y;
    err = sum(pred_id ~= GT_id) / length(y);
end