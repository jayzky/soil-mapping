from flask import Flask, jsonify, render_template, request, Response, send_file
import pandas as pd
import json
import os
import numpy as np
from models.soil_predictor import SoilPredictor
from utils.grid_generator import GridGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from werkzeug.utils import secure_filename
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
from werkzeug.middleware.proxy_fix import ProxyFix # 1. 导入 ProxyFix
from mplfonts import use_font # 导入 use_font

# 使用一个支持中文的字体，例如 'SimHei' (黑体)
# mplfonts 会尝试下载并使用该字体（如果本地没有）
use_font('Noto Sans CJK SC')

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 创建 Flask 应用，并指定模板目录
app = Flask(__name__, 
           template_folder=os.path.join(current_dir, 'templates'),
           static_folder=os.path.join(current_dir, 'static'))

# 2. 添加 ProxyFix 中间件
# 告诉应用去信任 Nginx 传来的 X-Forwarded-For, X-Forwarded-Proto 和 X-Script-Name (即 x_prefix)
app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)

# 配置文件上传
UPLOAD_FOLDER = os.path.join(current_dir, 'data', 'uploads')
ALLOWED_EXTENSIONS = {'csv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# 初始化模型和网格生成器
model = SoilPredictor()
grid_generator = GridGenerator(bounds=(29.05, 33.20, 108.21, 116.07))  # 湖北省边界
prediction_results = None

def load_data():
    """加载原始土壤数据"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data', 'raw', 'soil_data.csv')
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
        # 确保必要的列存在
        required_columns = ['city', 'latitude', 'longitude', 'soil_type', 'soil_texture', 'soil_color',
                          'elevation', 'slope', 'aspect', 'ndvi', 'precipitation', 'temperature']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("数据文件缺少必要的列")
        return df
    except Exception as e:
        print(f"加载数据失败: {str(e)}")
        return None

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@app.route('/api/sample_points')
def get_sample_points():
    """获取原始样本点数据，包括筛选所需的唯一值"""
    try:
        data = load_data()
        if data is None:
            return jsonify({'status': 'error', 'message': '无法加载数据'}), 500
        sample_points = data[['city', 'latitude', 'longitude', 'soil_type', 'soil_texture', 'soil_color']].to_dict('records')
        filter_options = {
            'cities': sorted(data['city'].unique().tolist()),
            'soil_types': sorted(data['soil_type'].unique().tolist()),
            'soil_textures': sorted(data['soil_texture'].unique().tolist())
        }
        return jsonify({
            'status': 'success',
            'points': sample_points,
            'filter_options': filter_options
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """训练模型"""
    try:
        # 加载数据
        data = load_data()
        if data is None:
            return jsonify({'status': 'error', 'message': '无法加载训练数据'}), 500
        
        # 检查数据完整性
        required_columns = ['elevation', 'slope', 'aspect', 'ndvi', 'precipitation', 'temperature', 'soil_type', 'soil_texture']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return jsonify({
                'status': 'error',
                'message': f'数据缺少必要的列: {", ".join(missing_columns)}'
            }), 400
        
        # 检查数据是否为空
        if len(data) == 0:
            return jsonify({'status': 'error', 'message': '训练数据为空'}), 400
        
        # 准备特征和标签
        X = data[['elevation', 'slope', 'aspect', 'ndvi', 'precipitation', 'temperature']]
        y = data[['soil_type', 'soil_texture']]
        
        # 检查数据有效性
        if X.isnull().any().any() or y.isnull().any().any():
            return jsonify({'status': 'error', 'message': '数据包含空值，请检查数据'}), 400
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 训练模型
        try:
            model.train(X_train, y_train)
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'模型训练失败: {str(e)}'}), 500
        
        # 获取评估指标和特征重要性
        try:
            metrics = model.evaluate(X_test, y_test)
            feature_importances = model.get_feature_importances()
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'模型评估失败: {str(e)}'}), 500
        
        # 保存测试集用于混淆矩阵
        global test_data
        test_data = {
            'X_test': X_test,
            'y_test': y_test
        }
        
        # 确保返回正确的评估指标格式
        evaluation_metrics = {
            'accuracy': float(metrics['soil_type']['accuracy']),
            'precision': float(metrics['soil_type']['precision']),
            'recall': float(metrics['soil_type']['recall']),
            'f1': float(metrics['soil_type']['f1_score']),
            'feature_names': ['高程 (m)', '坡度 (°)', '坡向 (°)', '植被指数', '降水量 (mm)', '温度 (°C)'],
            'feature_importances': [float(f"{importance:.4f}") for _, importance in feature_importances]
        }
        
        return jsonify({
            'status': 'success',
            'message': '模型训练成功',
            'metrics': evaluation_metrics
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'训练过程发生错误: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """生成预测并返回统计结果"""
    global prediction_results
    try:
        if not model.is_trained():
            return jsonify({
                'status': 'error',
                'message': '模型未训练，请先训练模型'
            }), 400
            
        # 生成网格点和特征
        grid_df = grid_generator.generate_grid()
        features = grid_generator.generate_features(grid_df)
        
        # 进行预测
        predictions = model.predict(features)
        
        # 合并结果
        prediction_results = pd.concat([
            grid_df[['latitude', 'longitude', 'city']],
            features,
            predictions
        ], axis=1)
        
        # 计算统计数据
        stats = calculate_prediction_stats(prediction_results)
        
        return jsonify({
            'status': 'success',
            'data': stats
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def calculate_prediction_stats(df):
    """计算预测结果的统计数据"""
    # 计算各城市的土壤类型和质地分布
    cities = sorted(df['city'].unique())
    soil_properties = {
        'categories': cities,
        'soil_types': {},  # 记录每个城市的土壤类型分布
        'soil_textures': {}  # 记录每个城市的土壤质地分布
    }
    
    for city in cities:
        city_data = df[df['city'] == city]
        # 计算土壤类型分布
        type_counts = city_data['soil_type'].value_counts()
        soil_properties['soil_types'][city] = type_counts.to_dict()
        
        # 计算土壤质地分布
        texture_counts = city_data['soil_texture'].value_counts()
        soil_properties['soil_textures'][city] = texture_counts.to_dict()
    
    # 计算环境因子的分布
    env_factors = {
        'elevation': {'name': '高程', 'unit': 'm', 'range': [0, 2000]},
        'slope': {'name': '坡度', 'unit': '°', 'range': [0, 90]},
        'aspect': {'name': '坡向', 'unit': '°', 'range': [0, 360]},
        'ndvi': {'name': '植被指数', 'unit': '', 'range': [0, 1]},
        'precipitation': {'name': '降水量', 'unit': 'mm', 'range': [0, 2000]},
        'temperature': {'name': '温度', 'unit': '°C', 'range': [-10, 40]}
    }
    
    env_stats = {}
    for factor, info in env_factors.items():
        if factor in df.columns:
            values = df[factor].dropna()
            env_stats[factor] = {
                'name': info['name'],
                'unit': info['unit'],
                'min': float(values.min()),
                'max': float(values.max()),
                'mean': float(values.mean()),
                'std': float(values.std()),
                'median': float(values.median()),
                'q1': float(values.quantile(0.25)),
                'q3': float(values.quantile(0.75)),
                'range': info['range']
            }
    
    return {
        'soil_properties': soil_properties,
        'environmental_factors': env_stats
    }

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload_sample_points', methods=['POST'])
def upload_sample_points():
    """处理样本点文件上传"""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': '未找到文件'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': '未选择文件'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'status': 'error', 'message': '不支持的文件格式'}), 400
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 验证文件格式
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            required_columns = ['city', 'latitude', 'longitude', 'soil_type', 'soil_texture', 'soil_color']
            if not all(col in df.columns for col in required_columns):
                raise ValueError('CSV文件缺少必要的列')
        except Exception as e:
            os.remove(filepath)  # 删除无效文件
            return jsonify({'status': 'error', 'message': f'文件格式错误: {str(e)}'}), 400
        
        return jsonify({'status': 'success', 'message': '文件上传成功'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def generate_soil_type_plot(df):
    """生成土壤类型分布图"""
    # 设置全局字体和样式
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'figure.figsize': (12, 8),
        'figure.dpi': 300
    })
    
    # 创建画布
    fig, ax = plt.subplots()
    
    # 土壤类型分布
    type_counts = df['soil_type'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(type_counts)))
    wedges, _, autotexts = ax.pie(type_counts.values, colors=colors,
                                 labels=[''] * len(type_counts),
                                 autopct='%1.1f%%',
                                 pctdistance=0.85)
    
    # 土壤类型图例（左侧）
    ax.legend(wedges, type_counts.index,
             title='土壤类型',
             bbox_to_anchor=(-0.3, 0.5),
             loc='center left',
             ncol=1,
             columnspacing=1,
             handlelength=1.5)
    ax.set_title('土壤类型分布', pad=20)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer

def generate_soil_texture_plot(df):
    """生成土壤质地分布图"""
    # 设置全局字体和样式
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'figure.figsize': (12, 8),
        'figure.dpi': 300
    })
    
    # 创建画布
    fig, ax = plt.subplots()
    
    # 土壤质地分布
    texture_counts = df['soil_texture'].value_counts()
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(texture_counts)))
    wedges, _, autotexts = ax.pie(texture_counts.values, colors=colors,
                                 labels=[''] * len(texture_counts),
                                 autopct='%1.1f%%',
                                 pctdistance=0.85)
    
    # 土壤质地图例（右侧）
    ax.legend(wedges, texture_counts.index,
             title='土壤质地',
             bbox_to_anchor=(1.3, 0.5),
             loc='center right',
             ncol=1,
             columnspacing=1,
             handlelength=1.5)
    ax.set_title('土壤质地分布', pad=20)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer

def generate_environmental_plots(df):
    """生成环境因子分布图表"""
    # 设置全局字体和样式
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'figure.figsize': (20, 15),
        'figure.dpi': 300
    })
    
    # 创建画布和子图
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
    
    # 环境因子分布
    factors = {
        'elevation': {'title': '高程分布', 'unit': 'm', 'color': '#2ecc71', 'bins': 20},
        'slope': {'title': '坡度分布', 'unit': '°', 'color': '#e74c3c', 'bins': 15},
        'aspect': {'title': '坡向分布', 'unit': '°', 'color': '#3498db', 'bins': 18},
        'ndvi': {'title': 'NDVI分布', 'unit': '', 'color': '#27ae60', 'bins': 20},
        'precipitation': {'title': '降水量分布', 'unit': 'mm', 'color': '#9b59b6', 'bins': 20},
        'temperature': {'title': '温度分布', 'unit': '℃', 'color': '#f1c40f', 'bins': 15}
    }
    
    for i, (factor, info) in enumerate(factors.items()):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        
        # 创建柱状图
        data = df[factor].dropna()
        n, bins, patches = ax.hist(data, bins=info['bins'], color=info['color'], alpha=0.7, edgecolor='black')
        
        # 添加均值线
        mean_val = data.mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_val:.2f}')
        
        # 添加中位数线
        median_val = data.median()
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'中位数: {median_val:.2f}')
        
        # 设置标题和标签
        title = f"{info['title']}"
        if info['unit']:
            title += f" ({info['unit']})"
        ax.set_title(title)
        
        # 设置x轴
        if factor == 'aspect':
            ax.set_xticks(np.arange(0, 361, 45))
        elif factor == 'ndvi':
            ax.set_xticks(np.arange(0, 1.1, 0.2))
        elif factor == 'precipitation':
            ax.set_xticks(np.arange(800, 1501, 100))
            ax.tick_params(axis='x', rotation=45)
        elif factor == 'temperature':
            ax.set_xticks(np.arange(10, 26, 2))
        elif factor == 'elevation':
            ax.tick_params(axis='x', rotation=45)
        
        ax.set_xlabel('')
        ax.set_ylabel('频数')
        ax.legend()
        
        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 添加统计信息文本框
        stats_text = f'最小值: {data.min():.2f}\n最大值: {data.max():.2f}\n标准差: {data.std():.2f}'
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer

def generate_spatial_plots(df):
    """生成地理空间分布图"""
    # 设置全局样式
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'figure.figsize': (20, 20),
        'figure.dpi': 300
    })
    
    # 创建GeoDataFrame
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=3857)
    
    # 创建画布和子图
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # 定义绘图配置
    plots = [
        {
            'column': 'soil_type',
            'title': '土壤类型空间分布',
            'categorical': True,
            'cmap': 'Set3',
            'legend_title': '土壤类型',
            'legend_ncol': 1
        },
        {
            'column': 'soil_texture',
            'title': '土壤质地空间分布',
            'categorical': True,
            'cmap': 'Pastel1',
            'legend_title': '土壤质地',
            'legend_ncol': 1
        },
        {
            'column': 'elevation',
            'title': '高程空间分布',
            'categorical': False,
            'cmap': 'viridis',
            'legend_title': '高程 (m)',
            'legend_ncol': None
        },
        {
            'column': 'ndvi',
            'title': 'NDVI空间分布',
            'categorical': False,
            'cmap': 'RdYlGn',
            'legend_title': 'NDVI',
            'legend_ncol': None
        }
    ]
    
    for (i, j), plot_config in zip([(0,0), (0,1), (1,0), (1,1)], plots):
        ax = axes[i, j]
        
        # 绘制地图
        if plot_config['categorical']:
            gdf.plot(
                column=plot_config['column'],
                ax=ax,
                legend=True,
                legend_kwds={
                    'title': plot_config['legend_title'],
                    'bbox_to_anchor': (1.3, 1.0),
                    'loc': 'upper right',
                    'ncol': plot_config['legend_ncol']
                },
                categorical=True,
                cmap=plot_config['cmap'],
                markersize=50,
                alpha=0.6
            )
        else:
            gdf.plot(
                column=plot_config['column'],
                ax=ax,
                legend=True,
                legend_kwds={
                    'label': plot_config['legend_title'],
                    'orientation': 'horizontal',
                    'pad': 0.05
                },
                cmap=plot_config['cmap'],
                markersize=50,
                alpha=0.6
            )
        
        # 添加底图
        ctx.add_basemap(
            ax,
            source=ctx.providers.CartoDB.Positron,
            attribution_size=8
        )
        
        # 设置标题和样式
        ax.set_title(plot_config['title'])
        ax.set_axis_off()
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer

def generate_kriging_plots(df):
    """使用克里金插值生成空间分布图"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        from pykrige.ok import OrdinaryKriging
        import numpy as np
        
        # 设置全局样式
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'legend.fontsize': 10,
            'figure.figsize': (20, 20),
            'figure.dpi': 300
        })
        
        # 创建画布和子图
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        
        # 定义要插值的属性
        attributes = [
            ('elevation', '高程 (m)', 'viridis'),
            ('ndvi', 'NDVI', 'RdYlGn'),
            ('precipitation', '降水量 (mm)', 'Blues'),
            ('temperature', '温度 (℃)', 'Reds')
        ]
        
        for (i, j), (attr, title, cmap) in zip([(0,0), (0,1), (1,0), (1,1)], attributes):
            ax = axes[i, j]
            
            # 准备数据
            x = df['longitude'].values
            y = df['latitude'].values
            z = df[attr].values
            
            # 创建网格
            grid_x = np.linspace(min(x), max(x), 100)
            grid_y = np.linspace(min(y), max(y), 100)
            
            # 执行克里金插值
            OK = OrdinaryKriging(
                x, y, z,
                variogram_model='linear',
                verbose=False,
                enable_plotting=False
            )
            
            z_pred, ss = OK.execute('grid', grid_x, grid_y)
            
            # 绘制插值结果
            im = ax.imshow(z_pred.T, origin='lower',
                          extent=[min(x), max(x), min(y), max(y)],
                          cmap=cmap)
            
            # 添加散点图
            ax.scatter(x, y, c='black', s=10, alpha=0.5)
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(title)
            
            # 设置标题和标签
            ax.set_title(f'{title}克里金插值图')
            ax.set_xlabel('经度')
            ax.set_ylabel('纬度')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
    except Exception as e:
        print(f"克里金插值生成失败: {str(e)}")
        return None

@app.route('/api/download')
def download_results():
    """下载预测结果"""
    try:
        if prediction_results is None:
            return jsonify({
                'status': 'error',
                'message': '尚未生成预测结果，请先进行预测'
            }), 400
            
        format = request.args.get('format', 'csv')
        
        if format == 'csv':
            csv_data = prediction_results.to_csv(index=False, encoding='utf-8')
            return Response(
                csv_data.encode('utf-8-sig'),
                mimetype='text/csv; charset=utf-8-sig',
                headers={
                    'Content-Disposition': 'attachment; filename=prediction_results.csv',
                    'Content-Type': 'text/csv; charset=utf-8-sig'
                }
            )
        elif format == 'json':
            return Response(
                prediction_results.to_json(orient='records', force_ascii=False).encode('utf-8'),
                mimetype='application/json; charset=utf-8',
                headers={
                    'Content-Disposition': 'attachment; filename=prediction_results.json',
                    'Content-Type': 'application/json; charset=utf-8'
                }
            )
        elif format == 'geojson':
            features = [{
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [row['longitude'], row['latitude']]
                },
                'properties': row.drop(['latitude', 'longitude']).to_dict()
            } for _, row in prediction_results.iterrows()]
            
            geojson = {
                'type': 'FeatureCollection',
                'features': features
            }
            
            return Response(
                json.dumps(geojson, ensure_ascii=False).encode('utf-8'),
                mimetype='application/geo+json; charset=utf-8',
                headers={
                    'Content-Disposition': 'attachment; filename=prediction_results.geojson',
                    'Content-Type': 'application/geo+json; charset=utf-8'
                }
            )
        elif format == 'soil_type':
            # 生成土壤类型分布图
            img_buffer = generate_soil_type_plot(prediction_results)
            return send_file(
                img_buffer,
                mimetype='image/png',
                as_attachment=True,
                download_name='soil_type_plot.png'
            )
        elif format == 'soil_texture':
            # 生成土壤质地分布图
            img_buffer = generate_soil_texture_plot(prediction_results)
            return send_file(
                img_buffer,
                mimetype='image/png',
                as_attachment=True,
                download_name='soil_texture_plot.png'
            )
        elif format == 'environmental':
            # 生成环境因子分布图
            img_buffer = generate_environmental_plots(prediction_results)
            return send_file(
                img_buffer,
                mimetype='image/png',
                as_attachment=True,
                download_name='environmental_plots.png'
            )
        elif format == 'kriging':
            # 生成克里金插值图
            img_buffer = generate_kriging_plots(prediction_results)
            if img_buffer:
                return send_file(
                    img_buffer,
                    mimetype='image/png',
                    as_attachment=True,
                    download_name='kriging_plots.png'
                )
            else:
                return jsonify({
                    'status': 'error',
                    'message': '克里金插值图生成失败，请确保已安装pykrige包'
                }), 500
        elif format == 'confusion_matrix':
            # 生成混淆矩阵图
            img_buffer = generate_confusion_matrix(prediction_results)
            return send_file(
                img_buffer,
                mimetype='image/png',
                as_attachment=True,
                download_name='confusion_matrix.png'
            )
        else:
            return jsonify({
                'status': 'error',
                'message': '不支持的格式'
            }), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def generate_confusion_matrix(df):
    """生成混淆矩阵图"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置全局样式
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'legend.fontsize': 10,
            'figure.figsize': (12, 10),
            'figure.dpi': 300
        })
        
        # 创建画布
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 计算混淆矩阵
        if 'test_data' in globals() and test_data is not None:
            y_true = test_data['y_test']['soil_type']
            y_pred = model.predict(test_data['X_test'])['soil_type']
        else:
            y_true = df['soil_type']
            y_pred = df['soil_type']
            
        cm = confusion_matrix(y_true, y_pred)
        
        # 获取类别标签
        labels = sorted(df['soil_type'].unique())
        
        # 计算百分比
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # 绘制混淆矩阵
        sns.heatmap(cm_percentage, 
                   annot=True, 
                   fmt='.1f', 
                   cmap='YlOrRd', 
                   ax=ax,
                   xticklabels=labels, 
                   yticklabels=labels, 
                   square=True,
                   cbar_kws={'label': '百分比 (%)', 'shrink': 0.8})
        
        # 设置标题和标签
        ax.set_title('土壤类型预测混淆矩阵 (%)', pad=20)
        ax.set_xlabel('预测类别', labelpad=10)
        ax.set_ylabel('真实类别', labelpad=10)
        
        # 调整坐标轴标签
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # 调整布局以确保所有标签可见
        plt.tight_layout()
        
        # 保存图片
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
    except Exception as e:
        print(f"混淆矩阵生成失败: {str(e)}")
        return None

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False) 