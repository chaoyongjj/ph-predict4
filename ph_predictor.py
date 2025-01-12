import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt


# 加载模型
@st.cache_resource
def load_model():
    with open('xgboost_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


model = load_model()

# 创建Streamlit应用
st.title('肺动脉高压院内死亡率预测')

# 创建输入字段
st.header('请输入患者信息：')
feature_names = ['SAPSII', 'HR', 'PO2', 'Lactate', 'RDW']  # 替换为你的特征名称
features = {}

for feature in feature_names:
    features[feature] = st.number_input(f'输入 {feature}:', value=0.0)

# 创建预测按钮
if st.button('预测'):
    # 将输入转换为DataFrame
    input_df = pd.DataFrame([features])

    # 进行预测
    prediction = model.predict_proba(input_df)[0][1]

    # 显示预测结果
    st.subheader('预测结果：')
    st.write(f'院内死亡概率: {prediction:.2%}')

    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    print(shap_values)
    # 绘制SHAP力图
    st.subheader('SHAP力图：')
#     fig, ax = plt.subplots()
    shap.force_plot(explainer.expected_value, shap_values[0], input_df,
                    feature_names=feature_names, matplotlib=True, show=False)
#     st.pyplot(fig)
    plt.savefig("force_plot.png", bbox_inches='tight', dpi=300)
    st.image("force_plot.png")
    # 绘制SHAP条形图
#     st.subheader('SHAP特征重要性：')
#     fig, ax = plt.subplots()
#     shap.summary_plot(shap_values, input_df, plot_type="bar", feature_names=feature_names, show=False)
#     st.pyplot(fig)

# 添加一些说明信息
st.markdown("""
### 使用说明：
1. 在上面的输入框中输入患者的相关信息。
2. 点击"预测"按钮获取预测结果。
3. 查看预测的院内死亡概率和SHAP解释图。

### 注意事项：
- 所有输入值应为数值型。
- 请确保输入的数据在合理范围内。
- SHAP力图显示了各个特征对预测结果的影响。
""")