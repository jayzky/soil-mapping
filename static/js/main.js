// 初始化地图
const map = L.map('map').setView([31.2, 112.3], 7);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

// 初始化图表
const soilPropertiesChart = echarts.init(document.getElementById('soilPropertiesChart'));
const environmentalFactorsChart = echarts.init(document.getElementById('environmentalFactorsChart'));
const modelMetricsChart = echarts.init(document.getElementById('modelMetrics'));
const featureImportanceChart = echarts.init(document.getElementById('featureImportance'));

// 存储预测结果
let predictions = null;
let markers = L.layerGroup().addTo(map);

// 加载数据
async function loadData() {
    try {
        const [features, labels] = await Promise.all([
            fetch('/api/features').then(res => res.json()),
            fetch('/api/labels').then(res => res.json())
        ]);
        
        // 确保数据中包含城市信息
        if (features && features.length > 0 && !features[0].city) {
            throw new Error('数据中缺少城市信息');
        }
        
        updateMap(features, labels);
        updateCharts(features, labels);
        populateSelects(features, labels);
    } catch (error) {
        console.error('加载数据失败:', error);
        showError('加载数据失败: ' + error.message);
    }
}

// 更新地图
function updateMap(features, labels, isPrediction = false) {
    // 清除现有标记
    markers.clearLayers();
    
    features.forEach((feature, i) => {
        const label = labels[i];
        const marker = L.marker([feature.latitude, feature.longitude]);
        
        marker.bindPopup(`
            <div class="popup-content">
                <h6>${feature.city || '预测点'}</h6>
                <p><b>经度:</b> ${feature.longitude.toFixed(4)}</p>
                <p><b>纬度:</b> ${feature.latitude.toFixed(4)}</p>
                <p><b>海拔:</b> ${feature.elevation.toFixed(1)}米</p>
                <p><b>坡度:</b> ${feature.slope.toFixed(1)}°</p>
                <p><b>NDVI:</b> ${feature.ndvi.toFixed(2)}</p>
                <p><b>降水量:</b> ${feature.precipitation.toFixed(0)}mm</p>
                <p><b>温度:</b> ${feature.temperature.toFixed(1)}°C</p>
                <p><b>土壤类型:</b> ${label.soil_type}</p>
                <p><b>土壤质地:</b> ${label.soil_texture}</p>
                <p><b>土壤颜色:</b> ${label.soil_color}</p>
                ${isPrediction ? '<p class="text-muted">* 此为预测结果</p>' : ''}
            </div>
        `);
        
        // 为预测结果使用不同的图标
        if (isPrediction) {
            marker.setIcon(L.divIcon({
                className: 'prediction-marker',
                html: `<div style="background-color: ${getSoilColor(label.soil_type)}; width: 8px; height: 8px; border-radius: 50%; border: 1px solid #666;"></div>`
            }));
        }
        
        markers.addLayer(marker);
    });
}

// 获取土壤颜色
function getSoilColor(soilType) {
    const colorMap = {
        '红壤': '#ff4444',
        '黄壤': '#ffaa00',
        '紫色土': '#9933cc',
        '水稻土': '#666666',
        '山地土': '#996633',
        '棕壤': '#8b4513',
        '黄棕壤': '#cd853f'
    };
    return colorMap[soilType] || '#888888';
}

// 更新图表
function updateCharts(features, labels) {
    // 土壤属性分布图
    const soilTypeCount = {};
    labels.forEach(label => {
        soilTypeCount[label.soil_type] = (soilTypeCount[label.soil_type] || 0) + 1;
    });

    soilPropertiesChart.setOption({
        title: { text: '土壤类型分布' },
        tooltip: { trigger: 'item' },
        legend: { orient: 'vertical', left: 'left' },
        series: [{
            type: 'pie',
            radius: '50%',
            data: Object.entries(soilTypeCount).map(([name, value]) => ({ name, value })),
            emphasis: {
                itemStyle: {
                    shadowBlur: 10,
                    shadowOffsetX: 0,
                    shadowColor: 'rgba(0, 0, 0, 0.5)'
                }
            }
        }]
    });

    // 环境因素分布图
    const environmentalData = features.map(f => ({
        elevation: f.elevation,
        slope: f.slope,
        precipitation: f.precipitation,
        temperature: f.temperature
    }));

    environmentalFactorsChart.setOption({
        title: { text: '环境因素分布' },
        tooltip: { trigger: 'axis' },
        legend: { data: ['高程', '坡度', '降水量', '温度'] },
        xAxis: { type: 'category', data: environmentalData.map((_, i) => `点${i + 1}`) },
        yAxis: { type: 'value' },
        series: [
            { name: '高程', type: 'line', data: environmentalData.map(d => d.elevation) },
            { name: '坡度', type: 'line', data: environmentalData.map(d => d.slope) },
            { name: '降水量', type: 'line', data: environmentalData.map(d => d.precipitation) },
            { name: '温度', type: 'line', data: environmentalData.map(d => d.temperature) }
        ]
    });
}

// 更新模型评估图表
function updateModelMetrics(metrics) {
    modelMetricsChart.setOption({
        title: { text: '模型评估指标' },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: ['准确率', '精确率', '召回率', 'F1分数'] },
        yAxis: { type: 'value', min: 0, max: 1 },
        series: [{
            type: 'bar',
            data: [
                metrics.accuracy,
                metrics.precision,
                metrics.recall,
                metrics.f1_score
            ],
            itemStyle: {
                color: function(params) {
                    const value = params.data;
                    return value >= 0.8 ? '#91cc75' : value >= 0.6 ? '#fac858' : '#ee6666';
                }
            }
        }]
    });

    featureImportanceChart.setOption({
        title: { text: '特征重要性' },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: metrics.feature_names },
        yAxis: { type: 'value', min: 0 },
        series: [{
            type: 'bar',
            data: metrics.feature_importances,
            itemStyle: {
                color: '#5470c6'
            }
        }]
    });
}

// 填充选择框
function populateSelects(features, labels) {
    const cities = [...new Set(features.map(f => f.city))];
    const soilTypes = [...new Set(labels.map(l => l.soil_type))];
    const soilTextures = [...new Set(labels.map(l => l.soil_texture))];

    populateSelect('citySelect', cities);
    populateSelect('soilTypeSelect', soilTypes);
    populateSelect('soilTextureSelect', soilTextures);
}

function populateSelect(selectId, options) {
    const select = document.getElementById(selectId);
    select.innerHTML = '<option value="">全部</option>';
    options.forEach(option => {
        const optionElement = document.createElement('option');
        optionElement.value = option;
        optionElement.textContent = option;
        select.appendChild(optionElement);
    });
}

// 显示错误信息
function showError(message) {
    const statusDiv = document.getElementById('modelStatus');
    statusDiv.style.display = 'block';
    statusDiv.className = 'alert alert-danger';
    statusDiv.textContent = message;
}

// 显示成功信息
function showSuccess(message) {
    const statusDiv = document.getElementById('modelStatus');
    statusDiv.style.display = 'block';
    statusDiv.className = 'alert alert-success';
    statusDiv.textContent = message;
}

// 显示状态信息
function showStatus(message, type) {
    const statusDiv = document.getElementById('modelStatus');
    statusDiv.style.display = 'block';
    statusDiv.className = `alert alert-${type}`;
    statusDiv.textContent = message;
}

// 模型训练
document.getElementById('trainModel').addEventListener('click', async () => {
    const statusDiv = document.getElementById('modelStatus');
    statusDiv.style.display = 'block';
    statusDiv.className = 'alert alert-info';
    statusDiv.textContent = '模型训练中...';

    try {
        const response = await fetch('/api/train', { method: 'POST' });
        const result = await response.json();
        
        if (result.status === 'success') {
            showSuccess('模型训练成功！');
            // 获取并更新模型评估指标
            const metricsResponse = await fetch('/api/model_metrics');
            const metrics = await metricsResponse.json();
            updateModelMetrics(metrics);
        } else {
            throw new Error(result.message);
        }
    } catch (error) {
        showError('训练失败: ' + error.message);
    }
});

// 生成预测
document.getElementById('predict').addEventListener('click', async () => {
    try {
        showStatus('正在生成预测...', 'info');
        
        // 先训练模型
        const trainResponse = await fetch('/api/train', { method: 'POST' });
        const trainResult = await trainResponse.json();
        
        if (trainResult.status !== 'success') {
            throw new Error('模型训练失败: ' + trainResult.message);
        }
        
        showStatus('模型训练成功，正在生成预测...', 'info');
        
        // 然后进行预测
        const response = await fetch('/api/predict', { method: 'POST' });
        const result = await response.json();
        
        if (result.status !== 'success') {
            throw new Error('预测失败: ' + result.message);
        }
        
        const predictions = result.data;
        
        if (predictions && predictions.length > 0) {
            // 更新地图
            updateMap(
                predictions.map(p => ({
                    latitude: p.latitude,
                    longitude: p.longitude,
                    elevation: p.elevation,
                    slope: p.slope,
                    aspect: p.aspect,
                    ndvi: p.ndvi,
                    precipitation: p.precipitation,
                    temperature: p.temperature,
                    city: p.city
                })),
                predictions.map(p => ({
                    soil_type: p.soil_type,
                    soil_texture: p.soil_texture,
                    soil_color: p.soil_color
                })),
                true  // isPrediction = true
            );
            
            showStatus(`成功生成 ${predictions.length} 个预测点`, 'success');
            
            // 更新图表
            updateCharts(predictions, predictions);
        } else {
            throw new Error('预测结果为空');
        }
    } catch (error) {
        console.error('预测失败:', error);
        showStatus('预测失败: ' + error.message, 'error');
    }
});

// 数据筛选
document.getElementById('filterData').addEventListener('click', async () => {
    const filters = {
        city: document.getElementById('citySelect').value,
        soil_type: document.getElementById('soilTypeSelect').value,
        soil_texture: document.getElementById('soilTextureSelect').value
    };

    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(filters)
        });
        const result = await response.json();
        
        if (result.features && result.features.length > 0) {
            updateMap(result.features, result.labels);
            updateCharts(result.features, result.labels);
            showSuccess(`找到 ${result.features.length} 条匹配记录`);
        } else {
            showError('没有找到匹配的记录');
        }
    } catch (error) {
        showError('筛选数据失败: ' + error.message);
    }
});

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', () => {
    loadData();
    window.addEventListener('resize', () => {
        soilPropertiesChart.resize();
        environmentalFactorsChart.resize();
        modelMetricsChart.resize();
        featureImportanceChart.resize();
    });
}); 