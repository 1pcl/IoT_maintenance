% 读取 JSON 文件
data = jsondecode(fileread("result.json"));

% 横坐标（迭代次数）
iters_ga = 1:length(data.ga_avg_cost);
iters_sa = length(data.ga_min_cost) + (1:length(data.sa_min_cost));  % 接在 GA 后面

%% 1. GA 平均代价曲线
figure;
plot(iters_ga, data.ga_avg_cost, 'b-', 'LineWidth', 2);
xlabel('Iteration', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Average Cost', 'FontSize', 20, 'FontWeight', 'bold');
set(gca, 'FontSize', 18, 'FontWeight', 'bold'); % 坐标轴刻度字体
grid on;

%% 2. GA 种群多样性曲线
figure;
plot(iters_ga, data.ga_diversity, 'm-', 'LineWidth', 2);
xlabel('Iteration', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Diversity', 'FontSize', 20, 'FontWeight', 'bold');
set(gca, 'FontSize', 18, 'FontWeight', 'bold');
grid on;

%% 3. GA + SA 最小代价对比曲线
figure;
plot(iters_ga, data.ga_min_cost, 'r-', 'LineWidth', 2); hold on;
plot(iters_sa, data.sa_min_cost, 'g-', 'LineWidth', 2);
xlabel('Iteration', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Minimum Cost', 'FontSize', 20, 'FontWeight', 'bold');
set(gca, 'FontSize', 18, 'FontWeight', 'bold');
legend('GA Minimum Cost','SA Minimum Cost', 'FontSize', 18, 'FontWeight', 'bold');
grid on;
