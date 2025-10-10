% Clear environment
clear; clc; close all;

% Read text file
filename = 'animal_en_ga_cost.txt'; % Make sure the filename is correct 
% filename = 'meter_cost.txt';

% Initialize variables
total_costs = [];
line_count = 0;

try
    % Open file
    fid = fopen(filename, 'r', 'n', 'UTF-8');
    
    if fid == -1
        error('Unable to open file, please check the filename and path.');
    end
    
    % Read file content
    while ~feof(fid)
        line = fgetl(fid);
        line_count = line_count + 1;
        
        % Skip headers and separator lines
        if line_count <= 2 || contains(line, '===') || contains(line, '---')
            continue;
        end
        
        % Skip average statistics section
        if contains(line, '平均值统计') || contains(line, 'Average Statistics')
            break;
        end
        
        % Parse data lines (only process lines containing numbers)
        if ~isempty(line) && ~all(isspace(line))
            % Use regular expressions to extract numbers
            numbers = regexp(line, '-?\d+\.?\d*', 'match');
            
            % Ensure there are enough data columns
            if length(numbers) >= 2
                % Second column is total cost (index 2, since first column is run count)
                cost = str2double(numbers{2});
                if ~isnan(cost)
                    total_costs = [total_costs; cost];
                end
            end
        end
    end
    
    % Close file
    fclose(fid);
    
    % Check if data was successfully read
    if isempty(total_costs)
        error('No valid data was read from the file.');
    end
    
    % Calculate statistics
    mean_cost = mean(total_costs);
    std_cost = std(total_costs);
    min_cost = min(total_costs);
    max_cost = max(total_costs);
    median_cost = median(total_costs);
    
    % Create figure
    figure('Position', [100, 100, 1000, 600]);
    
    % Create density histogram
    h = histogram(total_costs, 15, 'FaceColor', [0.2, 0.6, 0.8], ...
                 'EdgeColor', 'black', 'FaceAlpha', 0.7, ...
                 'Normalization', 'pdf'); % Set to probability density
    
    hold on;
    
    % Add statistical lines
    xline(mean_cost, 'r--', 'LineWidth', 2, 'FontSize', 16, 'Label', sprintf('Average: %.2f', mean_cost));
    xline(median_cost, 'g--', 'LineWidth', 2, 'FontSize', 16, 'Label', sprintf('Median: %.2f', median_cost));
    
    % Add normal distribution curve for reference
    x = linspace(min(total_costs), max(total_costs), 1000);
    y = normpdf(x, mean_cost, std_cost);
    plot(x, y, 'k-', 'LineWidth', 2, 'DisplayName', 'Normal Distribution');
    
    % Beautify the graph
    grid on;
    xlabel('Total Cost', 'FontSize', 16, 'FontWeight', 'bold');
    ylabel('Density', 'FontSize', 16, 'FontWeight', 'bold');
    set(gca, 'FontSize', 18, 'FontWeight', 'bold');
    legend('Cost Distribution', 'Average Cost', 'Median Cost', 'Normal Distribution', ...
       'Location', 'northwest', 'FontSize', 16, 'Box', 'on', 'LineWidth', 1);
    
    % Display statistical information in command window
    fprintf('=== File Reading Results ===\n');
    fprintf('Successfully read %d runs of data\n', length(total_costs));
    fprintf('Data range: Line 1 to Line %d\n', length(total_costs));
    fprintf('\n=== Total Cost Statistical Analysis ===\n');
    fprintf('Average: %.2f\n', mean_cost);
    fprintf('Standard Deviation: %.2f\n', std_cost);
    fprintf('Coefficient of Variation: %.4f\n', std_cost/mean_cost);
    fprintf('Minimum: %.2f (Run %d)\n', min_cost, find(total_costs == min_cost, 1));
    fprintf('Maximum: %.2f (Run %d)\n', max_cost, find(total_costs == max_cost, 1));
    fprintf('Range: %.2f\n', max_cost - min_cost);
    fprintf('Median: %.2f\n', median_cost);
    fprintf('Q1 (25%%): %.2f\n', quantile(total_costs, 0.25));
    fprintf('Q3 (75%%): %.2f\n', quantile(total_costs, 0.75));
    
    % Save graph
    saveas(gcf, 'Total_Cost_Analysis.png');
    fprintf('\nGraph saved as "Total_Cost_Analysis.png"\n');
    
catch ME
    % Error handling
    if fid ~= -1
        fclose(fid);
    end
    fprintf('Error: %s\n', ME.message);
end