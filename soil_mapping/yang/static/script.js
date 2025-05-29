// 初始化地图
let map = L.map('map').setView([39.9042, 116.3912], 13);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

// 处理表单提交
document.getElementById('queryForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const region = document.getElementById('region').value;
    const soilType = document.getElementById('soilType').value;
    const property = document.getElementById('property').value;
    
    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ region, soilType, property })
        });
        
        const data = await response.json();
        updateResults(data);
    } catch (error) {
        console.error('Error:', error);
    }
});

// 更新结果
function updateResults(data) {
    // 清除现有标记
    map.eachLayer((layer) => {
        if (layer instanceof L.Marker) {
            map.removeLayer(layer);
        }
    });
    
    // 更新表格
    const tbody = document.getElementById('resultsBody');
    tbody.innerHTML = '';
    
    data.forEach(item => {
        // 添加地图标记
        const marker = L.marker([item.latitude, item.longitude])
            .bindPopup(`
                <b>位置：</b>${item.location}<br>
                <b>土类：</b>${item.soil_type}<br>
                <b>pH值：</b>${item.ph}<br>
                <b>有机质含量：</b>${item.organic_matter}%<br>
                <b>氮含量：</b>${item.nitrogen}%
            `)
            .addTo(map);
        
        // 添加表格行
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${item.location}</td>
            <td>${item.soil_type}</td>
            <td>${item.ph}</td>
            <td>${item.organic_matter}%</td>
            <td>${item.nitrogen}%</td>
        `;
        tbody.appendChild(row);
    });
} 