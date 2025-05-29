# 土壤类型预测与可视化系统

这是一个基于机器学习的土壤类型预测与可视化系统，用于分析和预测湖北省的土壤类型分布。系统集成了数据可视化、机器学习预测和地理空间分析功能。

## 功能特点

- 土壤样本点数据可视化
- 机器学习模型训练与预测
- 土壤类型和质地分布分析
- 环境因素分析（高程、坡度、坡向等）
- 空间分布可视化
- 预测结果导出

## 系统要求

- Python 3.7+
- 现代浏览器（Chrome、Firefox、Edge等）

## 安装步骤

1. 克隆项目到本地：
```bash
git clone [项目地址]
cd soil_mapping
```

2. 创建并激活虚拟环境：
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. 安装依赖包：
```bash
pip install -r requirements.txt
```

## 项目结构

```
soil_mapping/
├── app.py              # 主应用文件
├── requirements.txt    # 依赖包列表
├── data/              # 数据目录
│   ├── raw/          # 原始数据
│   └── uploads/      # 上传文件存储
├── models/           # 模型相关代码
├── templates/        # HTML模板
├── static/          # 静态资源文件
└── utils/           # 工具函数
```

## 使用说明

1. 启动应用：
```bash
python app.py
```

2. 访问系统：
- 打开浏览器访问 `http://localhost:5000`

3. 主要功能：
   - 数据上传：支持上传CSV格式的土壤样本数据
   - 模型训练：使用样本数据训练预测模型
   - 预测分析：生成土壤类型和质地预测结果
   - 可视化：查看各类分析图表
   - 结果导出：下载预测结果和分析报告

## 数据格式要求

上传的CSV文件应包含以下字段：
- city：城市名称
- latitude：纬度
- longitude：经度
- soil_type：土壤类型
- soil_texture：土壤质地
- soil_color：土壤颜色
- elevation：高程
- slope：坡度
- aspect：坡向
- ndvi：植被指数
- precipitation：降水量
- temperature：温度

## 注意事项

1. 确保数据文件格式正确，且包含所有必要字段
2. 上传文件大小限制为16MB
3. 建议使用Chrome或Firefox浏览器访问系统
4. 首次使用需要先训练模型才能进行预测

## 常见问题

1. 如果页面无法显示，请检查：
   - 服务器是否正常启动
   - 浏览器控制台是否有错误信息
   - 端口5000是否被占用

2. 如果上传文件失败，请确认：
   - 文件格式是否为CSV
   - 文件大小是否超过限制
   - 文件是否包含所有必要字段
3. 复现此项目需要的gdal包安装可能比较复杂，需要在github上找到py相应的版本的whl文件通过pip install安装
## 技术支持

如有问题，请提交Issue或联系技术支持。
## 运行图片
![image](https://github.com/user-attachments/assets/69c422dd-7e80-45c0-93f3-48d4173fdc6f)
![image](https://github.com/user-attachments/assets/eab33854-d6b9-45c1-93eb-baee341425da)
![image](https://github.com/user-attachments/assets/47531761-0e32-4dc2-96e2-902081d2d6f1)
![image](https://github.com/user-attachments/assets/99a3c237-b479-4379-9d11-73f22d5ef139)



