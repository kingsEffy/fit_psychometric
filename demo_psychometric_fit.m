% demo_psychometric_fit.m
cd D:\zy\psychometric

T = readtable('psychometric_sample_trials_Marques2018_style.csv'); % from earlier
coh = T.signed_coherence;
y   = T.choice_right;

fit = fit_psychometric(coh, y);

% Report
fprintf('mu = %.2f, sigma = %.2f, lambda_L = %.3f, lambda_R = %.3f\n', ...
        fit.mu, fit.sigma, fit.lambda_left, fit.lambda_right);
fprintf('75%% threshold (lapse-corrected) c75 = %.2f\n', fit.c75);
fprintf('Slope at mu = %.4f per %%coh\n', fit.slope_at_mu);

% Plot binned proportions with fitted curve
[uc,~,ic] = unique(coh);
p_right   = accumarray(ic, y, [], @mean);
xgrid = linspace(min(coh), max(coh), 400);
figure; hold on;
plot(uc, p_right, 'ko', 'MarkerFaceColor',[0.2 0.2 0.2], 'DisplayName','Binned P(Right)');
plot(xgrid, fit.model(xgrid), 'r-', 'LineWidth',2, 'DisplayName','Fitted model');
yline(0.5,'k--'); xlabel('Signed coherence (%)'); ylabel('P(Right)');
legend('Location','best'); title('Psychometric — cumulative Gaussian with asymmetric lapses');
