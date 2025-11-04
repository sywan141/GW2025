% EATS (Equal Arrival Time Surface) plot in x–y plane
% Parameters
t_obs = 3.0;              % observer time [s]
Gamma = 300.0;            % Lorentz factor
c = 2.99792458e8;         % speed of light [m/s]
beta = sqrt(1 - 1/Gamma^2);

theta = linspace(0, pi, 200);

r = beta * c * t_obs ./ (1 - beta * cos(theta));
r_ls = r / c;  % radius in light-seconds
x_ls = r_ls .* cos(theta);
y_ls = r_ls .* sin(theta);

% Plot EATS
figure;
plot(x_ls, y_ls, 'b', 'LineWidth', 1.5); hold on;
plot(x_ls, -y_ls, 'b', 'LineWidth', 1.5); % symmetric lower half
plot(0, 0, 'ko', 'MarkerFaceColor', 'k'); % origin
xlabel('x/c (s)');
ylabel('y/c (s)');
% xscale('log');
title(sprintf('EATS for t_{obs}=%.1f s, \\Gamma=%.0f (\\beta=%.8f)', ...
      t_obs, Gamma, beta));
grid on;
axis equal;
xlim([0, max(x_ls)*1.05]);
ylim([-max(y_ls)*1.05, max(y_ls)*1.05]);