% 读取 JSON 文件
data = jsondecode(fileread("result.json"));

% 横坐标（迭代次数）
iters_ga = 1:length(data.avg_cost);

%% 1. GA 平均代价曲线
figure;
plot(iters_ga, data.avg_cost, 'b-', 'LineWidth', 2);
xlabel('Iteration', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Average Cost', 'FontSize', 20, 'FontWeight', 'bold');
set(gca, 'FontSize', 18, 'FontWeight', 'bold'); % 坐标轴刻度字体
grid on;

%% 2. GA 种群多样性曲线
figure;
plot(iters_ga, data.diversity, 'm-', 'LineWidth', 2);
xlabel('Iteration', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Diversity', 'FontSize', 20, 'FontWeight', 'bold');
set(gca, 'FontSize', 18, 'FontWeight', 'bold');
grid on;

