function compare_learning_rates(dataPath)
% === One-file validacija u MATLAB-u (TAE dataset) ===
% Poredi 3 strategije: Adaptive, Constant, Decrement + ROC (one-vs-rest).
% Poziv:
% compare_learning_rates('C:\...\neural-network-classification-master\tae.data')

    % --- provera argumenta ---
    assert(nargin==1 && (ischar(dataPath) || isstring(dataPath)), ...
        'Prosledi punu putanju do fajla, npr. compare_learning_rates(''C:\...\tae.data'')');

    % --- 1) učitavanje podataka (radi za .data/.dat/.csv) ---
    [X, y] = load_tae_numeric(dataPath);

    % --- 2) z-score normalizacija ---
    mu = mean(X,1); sig = std(X,[],1); sig(sig==0)=1;
    Xz = (X - mu)./sig;

    % --- 3) ciljevi kao one-hot ---
    T = full(ind2vec(y'));

    % --- 4) podela (70/15/15) ---
    trRatio = 0.7; valRatio = 0.15; teRatio = 0.15;

    % --- 5) parametri mreže ---
    epochs = 10000;
    hidden = [20];    % brz demo

    % ===== Constant LR =====
    netC = patternnet(hidden,'traingd');
    netC.divideParam.trainRatio = trRatio; netC.divideParam.valRatio = valRatio; netC.divideParam.testRatio = teRatio;
    netC.trainParam.lr = 0.01;
    netC.trainParam.epochs = epochs;
    netC.trainParam.max_fail = 12;
    netC.performParam.regularization = 0.0;
    [netC,trC] = train(netC, Xz', T);
    accC = 1 - trC.perf;   % proxy

    % ===== Adaptive LR =====
    netA = patternnet(hidden,'traingda');
    netA.divideParam.trainRatio = trRatio; netA.divideParam.valRatio = valRatio; netA.divideParam.testRatio = teRatio;
    netA.trainParam.lr = 0.01;
    netA.trainParam.epochs = epochs;
    netA.trainParam.max_fail = 12;
    [netA,trA] = train(netA, Xz', T);
    accA = 1 - trA.perf;   % proxy

    % ===== Decrement (viši start LR) =====
    netD = patternnet(hidden,'traingd');
    netD.divideParam.trainRatio = trRatio; netD.divideParam.valRatio = valRatio; netD.divideParam.testRatio = teRatio;
    netD.trainParam.lr = 0.05;
    netD.trainParam.epochs = epochs;
    netD.trainParam.max_fail = 12;
    [netD,trD] = train(netD, Xz', T);
    accD = 1 - trD.perf;   % proxy

    % --- 6) “prava” tačnost (na celom skupu; brza provera) ---
    ACCc = mean(vec2ind(netC(Xz'))' == y);
    ACCa = mean(vec2ind(netA(Xz'))' == y);
    ACCd = mean(vec2ind(netD(Xz'))' == y);
    fprintf('Final (whole data) ACC  |  Constant: %.3f  |  Adaptive: %.3f  |  Decrement: %.3f\n', ACCc, ACCa, ACCd);

    % --- 7) plot 3 krive kao u Pythonu ---
    figure('Name','Learning strategies (proxy accuracy curves)');
    plot(accA,'LineWidth',1.2); hold on;
    plot(accC,'LineWidth',1.2);
    plot(accD,'LineWidth',1.2);
    grid on; xlabel('epochs'); ylabel('acc proxy = 1 - perf');
    legend('Adaptive','Constant','Decrement','Location','northwest');
    title('Poređenje strategija učenja (brza vizuelna validacija)');

    % --- 8) ROC za najbolju mrežu (one-vs-rest) ---
    [~, idxBest] = max([ACCa, ACCc, ACCd]);
    nets = {netA, netC, netD};
    bestNet = nets{idxBest};
    plot_roc_tae(bestNet, Xz, y);

    % --- 9) Confusion 
        figure('Name','Confusion'); 
        confusionchart(confusionmat(y, vec2ind(bestNet(Xz'))'), string(1:3));
    catch
        disp('Confusion chart preskočen (nije dostupan Statistics & Machine Learning Toolbox).');
    end
end

% ===================== POMOĆNE FUNKCIJE (u istom fajlu) =====================

function [X, y] = load_tae_numeric(path)
    % Robustno čitanje TAE fajla (6 kolona: 5 atributa + klasa)
    if ~isfile(path), error('Fajl ne postoji: %s', path); end

    % Pokušaj CSV…
    try
        T = readtable(path, 'FileType','text', 'Delimiter',',', 'ReadVariableNames',false);
    catch
        % …pa whitespace
        T = readtable(path, 'FileType','text', 'Delimiter','\t', 'ReadVariableNames',false);
    end

    % Ako je i dalje malo kolona, probati auto detekciju
    if width(T) < 6
        opts = detectImportOptions(path); opts.ReadVariableNames=false;
        T = readtable(path, opts);
        if width(T) < 6
            error('Očekujem ≥6 kolona (5 atributa + klasa). Nađeno: %d', width(T));
        end
    end

    % Uzmi prvih 6 kolona
    T = T(:,1:6);

    % X: prvih 5 kolona (double)
    X = double(table2array(T(:,1:5)));

    % y: poslednja kolona (1/2/3 ili low/medium/high)
    yc = T{:,6};
    if iscell(yc) || isstring(yc) || ischar(yc)
        s = string(yc);
        y = zeros(numel(s),1);
        for i=1:numel(s)
            si = lower(strtrim(s(i)));
            if si=="low" || si=="1"
                y(i)=1;
            elseif si=="medium" || si=="2"
                y(i)=2;
            elseif si=="high" || si=="3"
                y(i)=3;
            else
                error('Nepoznata klasa: %s', s(i));
            end
        end
    else
        y = double(yc);
        if min(y)==0, y = y + 1; end   % 0/1/2 -> 1/2/3
    end
    y = round(y(:));
end

function plot_roc_tae(net, X, y)
    % ROC (one-vs-rest) za 3 klase
    scores = net(X');        % 3 x N
    scores = scores';        % N x 3

    classes = unique(y);
    figure('Name','ROC (one-vs-rest)'); hold on;
    legendNames = cell(numel(classes),1);
    for i = 1:numel(classes)
        pos = (y == classes(i));
        % perfcurve zahteva Statistics & Machine Learning Toolbox
        [fpr,tpr,~,AUC] = perfcurve(pos, scores(:,i), true);
        plot(fpr,tpr,'LineWidth',1.5);
        legendNames{i} = sprintf('Class %d (AUC=%.2f)', classes(i), AUC);
    end
    plot([0 1],[0 1],'k--');
    xlabel('False positive rate'); ylabel('True positive rate');
function compare_learning_rates(dataPath)
% === One-file validacija u MATLAB-u (TAE dataset) ===
% Poredi 3 strategije: Adaptive, Constant, Decrement + ROC (one-vs-rest).
% Poziv:
% compare_learning_rates('C:\...\neural-network-classification-master\tae.data')

    % --- provera argumenta ---
    assert(nargin==1 && (ischar(dataPath) || isstring(dataPath)), ...
        'Prosledi punu putanju do fajla, npr. compare_learning_rates(''C:\...\tae.data'')');

    % --- 1) učitavanje podataka (radi za .data/.dat/.csv) ---
    [X, y] = load_tae_numeric(dataPath);

    % --- 2) z-score normalizacija ---
    mu = mean(X,1); sig = std(X,[],1); sig(sig==0)=1;
    Xz = (X - mu)./sig;

    % --- 3) ciljevi kao one-hot ---
    T = full(ind2vec(y'));

    % --- 4) podela (70/15/15) ---
    trRatio = 0.7; valRatio = 0.15; teRatio = 0.15;

    % --- 5) parametri mreže ---
    epochs = 10000;
    hidden = [20];    % brz demo

    % ===== Constant LR =====
    netC = patternnet(hidden,'traingd');
    netC.divideParam.trainRatio = trRatio; netC.divideParam.valRatio = valRatio; netC.divideParam.testRatio = teRatio;
    netC.trainParam.lr = 0.01;
    netC.trainParam.epochs = epochs;
    netC.trainParam.max_fail = 12;
    netC.performParam.regularization = 0.0;
    [netC,trC] = train(netC, Xz', T);
    accC = 1 - trC.perf;   % proxy

    % ===== Adaptive LR =====
    netA = patternnet(hidden,'traingda');
    netA.divideParam.trainRatio = trRatio; netA.divideParam.valRatio = valRatio; netA.divideParam.testRatio = teRatio;
    netA.trainParam.lr = 0.01;
    netA.trainParam.epochs = epochs;
    netA.trainParam.max_fail = 12;
    [netA,trA] = train(netA, Xz', T);
    accA = 1 - trA.perf;   % proxy

    % ===== Decrement (viši start LR) =====
    netD = patternnet(hidden,'traingd');
    netD.divideParam.trainRatio = trRatio; netD.divideParam.valRatio = valRatio; netD.divideParam.testRatio = teRatio;
    netD.trainParam.lr = 0.05;
    netD.trainParam.epochs = epochs;
    netD.trainParam.max_fail = 12;
    [netD,trD] = train(netD, Xz', T);
    accD = 1 - trD.perf;   % proxy

    % --- 6) “prava” tačnost (na celom skupu; brza provera) ---
    ACCc = mean(vec2ind(netC(Xz'))' == y);
    ACCa = mean(vec2ind(netA(Xz'))' == y);
    ACCd = mean(vec2ind(netD(Xz'))' == y);
    fprintf('Final (whole data) ACC  |  Constant: %.3f  |  Adaptive: %.3f  |  Decrement: %.3f\n', ACCc, ACCa, ACCd);

    % --- 7) plot 3 krive kao u Pythonu ---
    figure('Name','Learning strategies (proxy accuracy curves)');
    plot(accA,'LineWidth',1.2); hold on;
    plot(accC,'LineWidth',1.2);
    plot(accD,'LineWidth',1.2);
    grid on; xlabel('epochs'); ylabel('acc proxy = 1 - perf');
    legend('Adaptive','Constant','Decrement','Location','northwest');
    title('Poređenje strategija učenja (brza vizuelna validacija)');

    % --- 8) ROC za najbolju mrežu (one-vs-rest) ---
    [~, idxBest] = max([ACCa, ACCc, ACCd]);
    nets = {netA, netC, netD};
    bestNet = nets{idxBest};
    plot_roc_tae(bestNet, Xz, y);

    % --- 9) Confusion
    try
        figure('Name','Confusion'); 
        confusionchart(confusionmat(y, vec2ind(bestNet(Xz'))'), string(1:3));
    catch
        disp('Confusion chart preskočen (nije dostupan Statistics & Machine Learning Toolbox).');
    end
end

% ===================== POMOĆNE FUNKCIJE (u istom fajlu) =====================

function [X, y] = load_tae_numeric(path)
    % Robustno čitanje TAE fajla (6 kolona: 5 atributa + klasa)
    if ~isfile(path), error('Fajl ne postoji: %s', path); end

    % Pokušaj CSV…
    try
        T = readtable(path, 'FileType','text', 'Delimiter',',', 'ReadVariableNames',false);
    catch
        % …pa whitespace
        T = readtable(path, 'FileType','text', 'Delimiter','\t', 'ReadVariableNames',false);
    end

    % Ako je i dalje malo kolona, probaj auto detekciju
    if width(T) < 6
        opts = detectImportOptions(path); opts.ReadVariableNames=false;
        T = readtable(path, opts);
        if width(T) < 6
            error('Očekujem ≥6 kolona (5 atributa + klasa). Nađeno: %d', width(T));
        end
    end

    % Uzmi prvih 6 kolona
    T = T(:,1:6);

    % X: prvih 5 kolona (double)
    X = double(table2array(T(:,1:5)));

    % y: poslednja kolona (1/2/3 ili low/medium/high)
    yc = T{:,6};
    if iscell(yc) || isstring(yc) || ischar(yc)
        s = string(yc);
        y = zeros(numel(s),1);
        for i=1:numel(s)
            si = lower(strtrim(s(i)));
            if si=="low" || si=="1"
                y(i)=1;
            elseif si=="medium" || si=="2"
                y(i)=2;
            elseif si=="high" || si=="3"
                y(i)=3;
            else
                error('Nepoznata klasa: %s', s(i));
            end
        end
    else
        y = double(yc);
        if min(y)==0, y = y + 1; end   % 0/1/2 -> 1/2/3
    end
    y = round(y(:));
end

function plot_roc_tae(net, X, y)
    % ROC (one-vs-rest) za 3 klase
    scores = net(X');        % 3 x N
    scores = scores';        % N x 3

    classes = unique(y);
    figure('Name','ROC (one-vs-rest)'); hold on;
    legendNames = cell(numel(classes),1);
    for i = 1:numel(classes)
        pos = (y == classes(i));
        % perfcurve zahteva Statistics & Machine Learning Toolbox
        [fpr,tpr,~,AUC] = perfcurve(pos, scores(:,i), true);
        plot(fpr,tpr,'LineWidth',1.5);
        legendNames{i} = sprintf('Class %d (AUC=%.2f)', classes(i), AUC);
    end
    plot([0 1],[0 1],'k--');
    xlabel('False positive rate'); ylabel('True positive rate');
    title('ROC krive (one-vs-rest)');
    legend(legendNames,'Location','SouthEast'); grid on;
end
    title('ROC krive (one-vs-rest)');
    legend(legendNames,'Location','SouthEast'); grid on;
end


