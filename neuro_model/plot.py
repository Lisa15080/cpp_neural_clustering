# plot.py - версия без matplotlib
import json
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

# Читаем JSON файлы
with open(os.path.join(base_dir, 'true_clusters.json')) as f:
    t = json.load(f)
with open(os.path.join(base_dir, 'predictions.json')) as f:
    p = json.load(f)

# Создаем HTML файл с интерактивным графиком
html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Визуализация классификации</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .plot-container { display: inline-block; width: 45%; margin: 10px; }
        h2 { text-align: center; }
    </style>
</head>
<body>
    <h1>Результаты классификации нейронной сети</h1>
    <div>
        <div class="plot-container">
            <h2>Истинные метки</h2>
            <div id="true_plot"></div>
        </div>
        <div class="plot-container">
            <h2>Предсказания сети</h2>
            <div id="pred_plot"></div>
        </div>
    </div>
    <script>
        // Данные из JSON
        const true_data = ''' + json.dumps(t) + ''';
        const pred_data = ''' + json.dumps(p) + ''';
        
        // Функция создания графика
        function createPlot(data, elementId, title) {
            const trace0 = {
                x: data.x.filter((_, i) => data.labels[i] === 0),
                y: data.y.filter((_, i) => data.labels[i] === 0),
                mode: 'markers',
                name: 'Класс 0',
                marker: { color: '#4472C4', size: 5 }
            };
            
            const trace1 = {
                x: data.x.filter((_, i) => data.labels[i] === 1),
                y: data.y.filter((_, i) => data.labels[i] === 1),
                mode: 'markers',
                name: 'Класс 1',
                marker: { color: '#ED7D31', size: 5 }
            };
            
            const layout = {
                title: title,
                xaxis: { title: 'X' },
                yaxis: { title: 'Y', scaleanchor: 'x', scaleratio: 1 },
                width: 550,
                height: 500
            };
            
            Plotly.newPlot(elementId, [trace0, trace1], layout);
        }
        
        // Создаем графики
        createPlot(true_data, 'true_plot', 'Истинные метки');
        createPlot(pred_data, 'pred_plot', 'Предсказания сети');
    </script>
</body>
</html>'''

# Сохраняем HTML файл
output_path = os.path.join(base_dir, 'visualization.html')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✅ Готово: {output_path}")
print("   Откройте файл visualization.html в браузере")