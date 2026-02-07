// 等待DOM加载完成
document.addEventListener('DOMContentLoaded', function() {
    // 初始化变量
    const predictionForm = document.getElementById('predictionForm');
    const resultCard = document.getElementById('resultCard');
    const errorAlert = document.getElementById('errorAlert');
    const submitBtn = document.getElementById('submitBtn');
    const buttonText = document.getElementById('buttonText');
    const spinner = document.getElementById('spinner');
    
    // 检查元素是否存在
    if (!predictionForm) {
        console.error('找不到表单元素 #predictionForm');
        return;
    }
    
    // 表单提交事件监听
    predictionForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // 验证表单
        if (!validateForm()) {
            return;
        }
        
        // 显示加载状态
        showLoading(true);
        
        // 隐藏之前的错误和结果
        hideError();
        hideResult();
        
        try {
            // 收集表单数据
            const formData = collectFormData();
            
            // 发送预测请求
            const response = await sendPredictionRequest(formData);
            
            if (response.success) {
                // 显示预测结果
                displayPredictionResult(response);
            } else {
                // 显示错误
                showError(response.error || '请求失败，请重试');
            }
        } catch (error) {
            console.error('请求错误:', error);
            showError(`网络错误: ${error.message}`);
        } finally {
            // 恢复按钮状态
            showLoading(false);
        }
    });
    
    // 表单验证函数
    function validateForm() {
        const waist = document.getElementById('waist').value;
        const height = document.getElementById('height').value;
        const weight = document.getElementById('weight').value;
        const age = document.getElementById('age').value;
        
        // 检查是否填写完整
        if (!waist || !height || !weight || !age) {
            showError('请填写所有必填字段');
            return false;
        }
        
        // 检查数值范围
        if (waist < 50 || waist > 200) {
            showError('腰围应在50-200cm之间');
            return false;
        }
        
        if (height < 100 || height > 250) {
            showError('身高应在100-250cm之间');
            return false;
        }
        
        if (weight < 30 || weight > 200) {
            showError('体重应在30-200kg之间');
            return false;
        }
        
        if (age < 18 || age > 100) {
            showError('年龄应在18-100岁之间');
            return false;
        }
        
        return true;
    }
    
    // 收集表单数据
    function collectFormData() {
        return {
            gender: document.querySelector('input[name="gender"]:checked').value,
            waist: parseFloat(document.getElementById('waist').value),
            height: parseFloat(document.getElementById('height').value),
            weight: parseFloat(document.getElementById('weight').value),
            age: parseInt(document.getElementById('age').value)
        };
    }
    
    // 发送预测请求
    async function sendPredictionRequest(formData) {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || `HTTP错误 ${response.status}`);
        }
        
        return data;
    }
    
    // 显示预测结果
    function displayPredictionResult(data) {
        const percentage = data.trunk_fat_percentage;
        const interpretation = data.interpretation;
        
        // 更新百分比显示
        document.getElementById('percentageDisplay').textContent = `${percentage}%`;
        
        // 更新风险等级
        const riskBadge = document.getElementById('riskBadge');
        riskBadge.textContent = interpretation.risk_level;
        
        // 根据风险等级设置样式
        if (interpretation.risk_level === '较低') {
            riskBadge.className = 'risk-badge bg-success-light';
        } else {
            riskBadge.className = 'risk-badge bg-danger-light';
        }
        
        // 更新建议文本
        document.getElementById('adviceText').textContent = interpretation.advice;
        document.getElementById('detailedAdvice').textContent = interpretation.detailed_advice;
        document.getElementById('cutoffNote').textContent = interpretation.cutoff_note;
        
        // 显示推荐建议列表
        const recommendationsList = document.getElementById('recommendationsList');
        recommendationsList.innerHTML = '';
        
        if (interpretation.recommendation && interpretation.recommendation.length > 0) {
            interpretation.recommendation.forEach(rec => {
                const li = document.createElement('li');
                li.className = 'list-group-item';
                li.innerHTML = `<i class="bi bi-check-circle me-2 text-success"></i>${rec}`;
                recommendationsList.appendChild(li);
            });
            document.getElementById('recommendationsContainer').style.display = 'block';
        } else {
            document.getElementById('recommendationsContainer').style.display = 'none';
        }
        
        // 显示结果卡片
        showResult();
        
        // 滚动到结果区域（平滑滚动）
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    
    // 显示加载状态
    function showLoading(loading) {
        if (loading) {
            buttonText.textContent = '评估中...';
            spinner.classList.remove('d-none');
            submitBtn.disabled = true;
            submitBtn.classList.add('disabled');
        } else {
            buttonText.textContent = '开始评估';
            spinner.classList.add('d-none');
            submitBtn.disabled = false;
            submitBtn.classList.remove('disabled');
        }
    }
    
    // 显示错误
    function showError(message) {
        errorAlert.textContent = message;
        errorAlert.classList.remove('d-none');
        
        // 3秒后自动隐藏错误
        setTimeout(() => {
            hideError();
        }, 5000);
    }
    
    // 隐藏错误
    function hideError() {
        errorAlert.classList.add('d-none');
    }
    
    // 显示结果
    function showResult() {
        resultCard.style.display = 'block';
    }
    
    // 隐藏结果
    function hideResult() {
        resultCard.style.display = 'none';
    }
    
    // 重置表单
    window.resetForm = function() {
        predictionForm.reset();
        hideResult();
        hideError();
        
        // 滚动到顶部
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };
    
    // 添加输入框实时验证
    const inputs = ['waist', 'height', 'weight', 'age'];
    inputs.forEach(inputId => {
        const input = document.getElementById(inputId);
        if (input) {
            input.addEventListener('input', function() {
                this.classList.remove('is-invalid');
            });
        }
    });
    
    // 页面加载完成提示
    console.log('躯干脂肪比例评估系统已加载完成');
});