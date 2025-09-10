function grid_search_ann(dataPath)
% Brzi 3D grid (n1,n2) -> validaciona tačnost za TAE dataset

    % --- 1) Ucitavanje podataka ---
    if nargin < 1 || isempty(dataPath)
        [f,p] = uigetfile({'*.data;*.dat;*.csv','TAE data'}, 'Izaberi TAE fajl');
        assert(f ~= 0, 'Nije izabran fajl.');
        dataPath = fullfile(p,f);
    end
    T = readmatrix(dataPath, 'FileType','text');

    X = T(:,1:5);   % ulazi
    y = T(:,6);     % klase (1..3)

    % --- 2) Normalizacija numerickih atributa ---
    mu = mean(X,1);
    sig = std(X,0,1); sig(sig==0) = 1;
    X = (X - mu) ./ sig;

    % --- 3) Podela train/val ---
    cv = cvpartition(y,'HoldOut',0.2);
    Xtr = X(training(cv),:);  ytr = y(training(cv));
    Xva = X(test(cv),:);      yva = y(test(cv));

    % --- 4) Grid ---
    n1 = 5:5:50;
    n2 = 5:5:50;
    ACC = zeros(numel(n1), numel(n2));

    rng(7);
    for i = 1:numel(n1)
        for j = 1:numel(n2)
            targets = full(ind2vec(ytr')); 
            net = patternnet([n1(i) n2(j)],'trainscg');
            net.divideMode = 'none'; 
            net.trainParam.epochs = 100;
            net.trainParam.showWindow = false;
            net = train(net, Xtr', targets);
            outputs = net(Xva');
            yhat = vec2ind(outputs)';
            ACC(i,j) = mean(yhat == yva);
        end
    end

    % --- 5) Nacrtaj 3D površ ---
    [N1,N2] = meshgrid(n1,n2);
    figure;
    surf(N1,N2,ACC'); 
    shading interp; colormap parula
    xlabel('1st layer neurons'); ylabel('2nd layer neurons'); zlabel('Val ACC');
    title(sprintf('Max ACC = %.3f', max(ACC(:))));
end
