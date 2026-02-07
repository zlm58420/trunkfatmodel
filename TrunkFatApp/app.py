from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import tempfile

app = Flask(__name__)

def fix_xgboost_model_attributes(model):
    """彻底修复XGBoost模型（重写get_params避免gpu_id报错）"""
    print("🔧 修复模型属性...")
    
    try:
        # 核心修复：重写模型的get_params方法，过滤掉gpu_id等不存在的参数
        original_get_params = model.get_params
        
        def custom_get_params(deep=True):
            """自定义get_params，过滤掉GPU相关参数"""
            params = original_get_params(deep=deep)
            # 移除所有GPU相关的参数键，避免访问不存在的属性
            gpu_params = ['gpu_id', 'n_gpus', 'device']
            for key in gpu_params:
                if key in params:
                    del params[key]
            # 强制设置CPU相关参数
            params['predictor'] = 'cpu_predictor'
            params['tree_method'] = 'hist'
            return params
        
        # 替换模型的get_params方法
        model.get_params = custom_get_params
        print("  ✅ 重写get_params方法，过滤GPU参数")
        
        # 处理存在的属性（只操作确实存在的）
        safe_attrs = ['tree_method', 'predictor', 'device']
        for attr in safe_attrs:
            if hasattr(model, attr):
                try:
                    if attr == 'tree_method':
                        setattr(model, attr, 'hist')
                    elif attr == 'predictor':
                        setattr(model, attr, 'cpu_predictor')
                    elif attr == 'device':
                        setattr(model, attr, 'cpu')
                    print(f"  设置 model.{attr} = {getattr(model, attr)}")
                except:
                    pass
        
        # 修复内部Booster
        if hasattr(model, '_Booster'):
            booster = model._Booster
            try:
                booster.set_param({'predictor': 'cpu_predictor'})
                print(f"  设置booster参数: predictor='cpu_predictor'")
            except:
                pass
        
        print("✅ 模型修复完成")
        return model
        
    except Exception as e:
        print(f"⚠️ 模型修复过程中出现错误: {e}")
        return model

def load_model():
    try:
        # 确保模型文件路径正确
        model_path = 'model/simplified_xgboost_tuned.pkl'
        if not os.path.exists(model_path):
            model_path = 'simplified_xgboost_tuned.pkl'
            
        print(f"📂 正在使用 joblib 加载模型: {model_path}")
        
        # 使用 joblib 加载模型
        model = joblib.load(model_path)
        
        # 优先修复get_params（核心！）
        model = fix_xgboost_model_attributes(model)
        
        print(f"✅ 模型加载成功!")
        print(f"📊 模型类型: {type(model)}")
        
        # 测试预测（现在可以安全执行）
        try:
            print("🧪 测试模型预测...")
            test_input = np.array([[1, 85.0, 175.0, 72.0, 45]])
            test_prediction = model.predict(test_input)
            print(f"✅ 模型测试预测成功: {test_prediction[0]:.2f}%")
                
        except Exception as test_error:
            print(f"⚠️ 模型测试失败: {test_error}")
            # 尝试深度修复（不依赖get_params）
            model = deep_fix_xgboost_model(model)
            # 重新测试
            test_prediction = model.predict(test_input)
            print(f"✅ 深度修复后测试预测成功: {test_prediction[0]:.2f}%")
        
        return model
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def deep_fix_xgboost_model(model):
    """深度修复：直接重建模型，不依赖原始参数"""
    print("🔧 深度修复模型...")
    
    try:
        import xgboost as xgb
        
        model_type = str(type(model))
        print(f"  模型类型: {model_type}")
        
        if 'XGBRegressor' in model_type:
            print("  创建新的XGBRegressor（纯CPU模式）...")
            
            # 关键：不调用model.get_params()，直接用默认CPU参数构建
            new_model = xgb.XGBRegressor(
                predictor='cpu_predictor',
                tree_method='hist',
                n_jobs=1,
                random_state=42
            )
            
            # 移植Booster（如果存在）
            if hasattr(model, '_Booster'):
                try:
                    with tempfile.NamedTemporaryFile(suffix='.model', delete=False) as tmp:
                        tmp_path = tmp.name
                        model._Booster.save_model(tmp_path)
                    
                    # 用虚拟数据拟合（仅为初始化）
                    X_dummy = np.random.rand(10, 5)
                    y_dummy = np.random.rand(10)
                    new_model.fit(X_dummy, y_dummy, verbose=False)
                    
                    # 加载原始booster
                    new_model._Booster.load_model(tmp_path)
                    os.unlink(tmp_path)
                    
                    print("  ✅ Booster移植成功")
                    return new_model
                    
                except Exception as e:
                    print(f"  ❌ Booster移植失败: {e}")
        
        return model
        
    except Exception as e:
        print(f"  ❌ 深度修复失败: {e}")
        return model

# 加载模型
model = load_model()

# 特征顺序必须与训练时一致
FEATURE_ORDER = ['Female', 'Waist', 'Height', 'Weight', 'Age']

@app.route('/')
def index():
    """渲染主页面"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """处理预测请求"""
    global model
    try:
        # 获取JSON数据
        data = request.get_json()
        
        # 验证必需字段
        required_fields = ['gender', 'waist', 'height', 'weight', 'age']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'缺少必要字段: {field}'}), 400
        
        # 验证数值有效性
        try:
            waist = float(data['waist'])
            height = float(data['height'])
            weight = float(data['weight'])
            age = float(data['age'])
            
            # 简单范围检查
            if not (50 <= waist <= 200):
                return jsonify({'error': '腰围应在50-200cm之间'}), 400
            if not (100 <= height <= 250):
                return jsonify({'error': '身高应在100-250cm之间'}), 400
            if not (30 <= weight <= 200):
                return jsonify({'error': '体重应在30-200kg之间'}), 400
            if not (18 <= age <= 100):
                return jsonify({'error': '年龄应在18-100岁之间'}), 400
                
        except ValueError:
            return jsonify({'error': '请输入有效的数值'}), 400
        
        # 数据预处理
        features = {
            'Female': 1 if data['gender'].lower() == 'female' else 0,
            'Waist': waist,
            'Height': height,
            'Weight': weight,
            'Age': age
        }
        
        # 创建特征数组
        feature_array = np.array([[features[feature] for feature in FEATURE_ORDER]])
        
        # 预测
        if model is None:
            return jsonify({'error': '模型未加载，请检查服务器配置'}), 500
        
        print(f"🔍 预测输入特征: {features}")
        
        try:
            prediction = model.predict(feature_array)[0]
            print(f"📈 原始预测值: {prediction}")
        except Exception as predict_error:
            error_msg = str(predict_error)
            print(f"❌ 模型预测失败: {error_msg}")
            # 紧急修复：重新加载模型
            model = load_model()
            if model is None:
                return jsonify({'error': '模型修复失败'}), 500
            prediction = model.predict(feature_array)[0]
            print(f"📈 紧急修复后预测值: {prediction}")
        
        # 限制预测范围在合理区间
        prediction = max(5.0, min(50.0, float(prediction)))
        prediction_rounded = round(prediction, 1)
        
        print(f"✅ 最终预测值: {prediction_rounded}%")
        
        return jsonify({
            'success': True,
            'trunk_fat_percentage': prediction_rounded,
            'interpretation': get_interpretation(prediction_rounded)
        })
        
    except Exception as e:
        print(f"❌ 预测错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

def get_interpretation(percentage):
    """根据预测结果提供健康解读"""
    if percentage < 28.6:
        risk = "较低"
        advice = "您的躯干脂肪比例在健康范围内。继续保持均衡饮食和规律运动。"
    else:
        risk = "较高"
        advice = "您的躯干脂肪比例提示代谢性疾病风险增高。建议咨询医生，调整饮食结构并增加有氧运动。"
    
    detailed_advice = ""
    if percentage < 20:
        detailed_advice = "优秀！您的身体成分非常健康。"
    elif percentage < 25:
        detailed_advice = "良好！继续保持当前的生活方式。"
    elif percentage < 28.6:
        detailed_advice = "注意！接近风险临界值，建议定期监测。"
    elif percentage < 35:
        detailed_advice = "关注！建议进行详细的身体成分分析，并制定改善计划。"
    else:
        detailed_advice = "重要提示！强烈建议寻求专业医疗指导。"
    
    return {
        'risk_level': risk,
        'advice': advice,
        'detailed_advice': detailed_advice,
        'cutoff_note': f"根据临床研究，躯干脂肪比例 ≥ 28.6% 被视为代谢性疾病的风险临界值。您的结果是 {percentage:.1f}%。",
        'recommendation': _get_recommendation(percentage)
    }

def _get_recommendation(percentage):
    """根据脂肪比例提供个性化建议"""
    recommendations = []
    if percentage >= 28.6:
        recommendations.append("增加有氧运动频率，每周至少150分钟中等强度运动")
        recommendations.append("减少精制碳水化合物和饱和脂肪的摄入")
        recommendations.append("增加膳食纤维和优质蛋白质比例")
        recommendations.append("考虑定期监测空腹血糖和血脂")
    if percentage >= 35:
        recommendations.append("强烈建议进行医学营养治疗咨询")
        recommendations.append("考虑进行口服葡萄糖耐量试验")
    if percentage < 25:
        recommendations.append("继续保持当前的健康生活习惯")
        recommendations.append("定期进行身体成分监测")
    return recommendations

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy' if model is not None else 'model_not_loaded',
        'model_loaded': model is not None,
        'features': FEATURE_ORDER
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 启动 Flask 应用，端口: {port}")
    app.run(host='0.0.0.0', port=port, debug=True)