cd D:\zy\psychometric\

% Load the sample data
S   = load('svm_sample_neural_meanResponses.mat');
X0  = S.features;                    % trials x neurons
yLR = double(S.choice_right_label);  % 1=Right/Temporal, 0=Left/Nasal
y    = 2*yLR - 1;                    % {-1,+1}
y    = y(:);
coh  = S.coherence(:);               % 16,40,100
dir  = y;                            % {-1,+1}

% Group labels for stratification: 6 stimulus types = coherence × direction
% Map to integers (e.g., 16_left, 16_right, 40_left, ...):
uC = unique(coh);   % [16 40 100]
grp = zeros(size(y));
for i=1:numel(uC)
    grp(coh==uC(i) & dir==-1) = 2*i-1;  % nasal
    grp(coh==uC(i) & dir==+1) = 2*i;    % temporal
end

rng(1)   % reproducible

Ns       = [1 2 5 10 20 50 80 100];       % no zero
repeats  = 60;                            % use more if you like
Cs       = uC;                            % report per coherence
acc      = nan(numel(Cs), numel(Ns));     % mean across repeats

for ni = 1:numel(Ns)
    N  = Ns(ni);
    accRep = zeros(numel(Cs), repeats);

    for r = 1:repeats
        % pick N neurons at random
        sel = randperm(size(X0,2), N);

        % ---------- stratified split by stimulus type ----------
        teIdx = [];
        for g = 1:max(grp)
            idxg = find(grp==g);
            nTe  = max(1, round(0.25*numel(idxg)));  % 25% test per type
            teIdx = [teIdx; idxg(randperm(numel(idxg), nTe))]; %#ok<AGROW>
        end
        trIdx = setdiff((1:numel(y))', teIdx);

        % standardise on the training set (per neuron)
        mu = mean(X0(trIdx, sel), 1);
        sd = std( X0(trIdx, sel), [], 1) + 1e-6;
        Xtr = (X0(trIdx, sel) - mu) ./ sd;
        Xte = (X0(teIdx,  sel) - mu) ./ sd;

        ytr = y(trIdx);
        yte = y(teIdx);

        % sanity: two classes present in both sets?
        if numel(unique(ytr))<2 || numel(unique(yte))<2
            % re-draw this repeat (very rare with stratification)
            r = r - 1;  continue
        end

        % linear SVM
        M = fitcsvm(Xtr, ytr, 'KernelFunction','linear', ...
                    'BoxConstraint',1, 'Standardize',false, ...
                    'ClassNames',[-1 1]);

        yhat = predict(M, Xte);
        yhat = yhat(:); yte = yte(:);   % avoid implicit expansion

        % accuracy per coherence (computed on THEIR test trials)
        for ci = 1:numel(Cs)
            m = (coh(teIdx)==Cs(ci));
            accRep(ci, r) = mean(yhat(m) == yte(m));
        end
    end

    acc(:, ni) = mean(accRep, 2);
end

% ---- plot like Fig. 6A (without behavioural line) ----
figure; hold on
plot(Ns, 100*acc(1,:), '-o', 'DisplayName','16%');
plot(Ns, 100*acc(2,:), '-o', 'DisplayName','40%');
plot(Ns, 100*acc(3,:), '-o', 'DisplayName','100%');
yline(50, '--'); xlabel('# neurons'); ylabel('Decoder accuracy (%)');
legend('Location','southeast'); title('Linear SVM decoder (mixed-coherence training)');


%%

% S = load('svm_sample_neural_meanResponses.mat');
% X = S.features;                    % trials x neurons
% y = double(S.choice_right_label);  % 1 = Right/Temporal, 0 = Left/Nasal
% coh = S.coherence;                 % 16, 40, 100
% 
% Ns = [1 2 5 10 20 50 80 100];
% Cs = [16 40 100];
% repeats = 50;
% acc = nan(numel(Cs), numel(Ns));
% 
% for ci = 1:numel(Cs)
%     mask = coh == Cs(ci);
%     Xc = X(mask,:);  yc = y(mask);
%     for ni = 1:numel(Ns)
%         N = Ns(ni);
%         tmp = nan(repeats,1);
%         for r = 1:repeats
%             sel = randperm(size(Xc,2), N);                % pick N neurons
%             % split trials 75/25
%             idx = randperm(size(Xc,1)); nt = floor(0.25*numel(idx));
%             te = idx(1:nt); tr = idx(nt+1:end);
%             mu = mean(Xc(tr,sel),1); sd = std(Xc(tr,sel),[],1)+1e-6; % standardise
%             Xtr = (Xc(tr,sel)-mu)./sd;  Xte = (Xc(te,sel)-mu)./sd;
% 
%             M = fitcsvm(Xtr, 2*yc(tr)-1, 'KernelFunction','linear', ...
%                         'BoxConstraint',1, 'Standardize', false, 'ClassNames',[-1 1]);
%             yhat = predict(M, Xte);
%             yte   = 2*yc(te) - 1;   yte = yte(:);
%             hits   = sum(yhat(:) == yte(:));
%             total  = numel(yte);
%             acc    = hits / total;
%             tmp(r) = acc;                 % Fig 6A-style
%             % % ...
%             % acc_t(f) = acc;               % time-resolved
% 
%         end
%         acc(ci,ni) = mean(tmp);  % decoder accuracy (fraction correct)
%     end
% end
% 
% % Plot like Fig. 6A (no behaviour line here)
% figure; hold on;
% plot(Ns, 100*acc(1,:), '-o'); plot(Ns, 100*acc(2,:), '-o'); plot(Ns, 100*acc(3,:), '-o');
% yline(50,'--'); xlabel('# neurons'); ylabel('Decoder accuracy (%)');
% legend('16%','40%','100%','Location','southeast'); title('Linear SVM decoder (simulated)');
