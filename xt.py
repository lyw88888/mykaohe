import streamlit as st  # Web应用框架
import pandas as pd     # 数据处理
import numpy as np      # 数值计算
import matplotlib.pyplot as plt  # 静态可视化
import plotly.express as px     # 交互式可视化
import pickle           # 模型序列化/反序列化


# 基础配置：修复中文乱码问题

# 设置中文字体，解决matplotlib图表中文显示乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# 解决负号显示异常问题
plt.rcParams['axes.unicode_minus'] = False


# 页面全局配置

# 设置页面标题、布局（宽屏）、图标
st.set_page_config(page_title="学生成绩分析与预测系统", layout="wide", page_icon="📊")

# 全局变量与数据加载

# 数据文件和模型文件路径
DATA_PATH = "student_data_adjusted_rounded.csv"  # 学生成绩数据文件
MODEL_PATH = "model.pkl"                         # 训练好的预测模型文件

# 加载学生成绩数据
try:
    df = pd.read_csv(DATA_PATH)  # 读取CSV数据文件
except Exception as e:
    # 数据加载失败时显示错误信息并停止程序
    st.error(f"❌ 无法加载数据文件: {e}")
    st.stop()


# 自定义CSS样式配置

custom_css = """
<style>
/* 按钮样式定制 */
.stButton>button {
    background-color: #ff4b4b;    /* 按钮背景色 */
    color: white;                 /* 按钮文字颜色 */
    border: none;                 /* 去掉边框 */
    padding: 4px 10px;            /* 内边距 */
    border-radius: 2px;           /* 圆角 */
    font-size: 10px;              /* 字体大小 */
}
/* 按钮hover效果 */
.stButton>button:hover {
    background-color: #e53935;    /* 鼠标悬浮时背景色 */
}
/* 标题样式调整 */
h1, h2, h3, h4, h5, h6 {
    margin-top: 0.5rem;           /* 上边距 */
    margin-bottom: 0.5rem;        /* 下边距 */
}
/* 数据表格字体大小 */
.stDataFrame {
    font-size: 11px;              /* 表格字体大小 */
}
/* 图片居中样式类 */
.center-image {
    display: flex;                /* 弹性布局 */
    justify-content: center;      /* 水平居中 */
    align-items: center;          /* 垂直居中 */
}
</style>
"""
# 应用自定义CSS样式
st.markdown(custom_css, unsafe_allow_html=True)

# 侧边栏导航菜单

with st.sidebar:  # 侧边栏容器
    st.title("🧭 导航菜单")  # 侧边栏标题
    # 单选按钮实现页面切换
    page = st.radio(
        "选择页面",                # 选项标题
        ["项目介绍", "专业数据分析", "成绩预测"],  # 可选页面
        help="选择要查看的功能页面"  # 提示信息
    )


# 页面1：项目介绍页

if page == "项目介绍":
    st.title("🎓 学生成绩分析与预测系统")  # 页面主标题
    
    # 项目简介
    st.markdown("""
    本项目是一个基于 Streamlit 的学生学业表现分析平台，通过数据可视化和机器学习技术，
    帮助教育工作者和学生深入了解学业表现，并预测期末考试成绩。
    """)

    # 布局：主内容区(3/4) + 图片预览区(1/4)
    col_main, col_sidebar = st.columns([3, 1])
    
    # 图片预览区（右侧）
    with col_sidebar:
        st.markdown("### 图片预览")  # 子标题
        
        # 初始化session state：记录当前显示的图片索引（避免页面刷新丢失状态）
        if 'current_img_index' not in st.session_state:
            st.session_state.current_img_index = 0
        
        # 图片列表和对应说明
        img_list = ["1.png", "2.png"]  # 图片文件路径
        img_captions = ["学生数据可视化示意图", "系统架构图"]  # 图片说明
             
        
        # 获取当前要显示的图片和说明
        current_img = img_list[st.session_state.current_img_index]
        current_caption = img_captions[st.session_state.current_img_index]
        
        try:
            # 显示图片（固定宽度200px）
            st.image(current_img, caption=current_caption, width=200)
        except FileNotFoundError:
            # 图片文件不存在时显示警告
            st.warning(f"图片文件 {current_img} 未找到")
            # 显示占位符图片
            st.image("https://via.placeholder.com/200x150?text=图片未找到", caption="图片加载失败", width=200)

             # 图片切换按钮布局（上一页/下一页）
        col_prev, col_next = st.columns([1, 1])
        with col_prev:
            # 上一页按钮：索引减1，取模实现循环
            if st.button("◀", key="prev_img"):
                st.session_state.current_img_index = (st.session_state.current_img_index - 1) % len(img_list)
                 
        with col_next:
            # 下一页按钮：索引加1，取模实现循环
            if st.button("▶", key="next_img"):
                st.session_state.current_img_index = (st.session_state.current_img_index + 1) % len(img_list)

    # 主内容区（左侧）
    with col_main:
        # 项目目标模块
        st.header("🎯 项目目标")
        # 三列布局展示目标
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**目标一**\n- 分析影响成绩的关键因素\n- 探索成绩相关性\n- 提供教学支持决策")
        with col2:
            st.markdown("**目标二**\n- 可视化展示各专业数据\n- 性别比例分析\n- 学习行为对比")
        with col3:
            st.markdown("**目标三**\n- 基于机器学习模型预测\n- 个性化成绩预测\n- 及时干预建议")

        # 技术架构模块
        st.header("🛠️ 技术架构")
        # 四列布局展示技术栈
        tech_cols = st.columns(4)
        tech_cols[0].markdown("**前端框架**\nStreamlit")
        tech_cols[1].markdown("**数据处理**\nPandas\nNumPy")
        tech_cols[2].markdown("**可视化**\nMatplotlib\nPlotly")
        tech_cols[3].markdown("**机器学习**\nScikit-learn")


# 页面2：专业数据分析页

elif page == "专业数据分析":
    st.title("📊 专业数据分析")  # 页面主标题
    st.markdown("#### 专业数据可视化分析")  # 子标题

    # 1. 各专业男女性别比例分析
    st.subheader("1. 各专业男女性别比例")
    # 按专业和性别分组统计人数，缺失值填充为0
    gender_count = df.groupby(["专业", "性别"]).size().unstack(fill_value=0)
    
    # 布局：图表区(3/4) + 表格区(1/4)
    col_chart, col_table = st.columns([3, 1])
    with col_chart:
        # 创建图表（缩小尺寸：5.5x2.8）
        fig1, ax1 = plt.subplots(figsize=(5.5, 2.8))
        # 绘制柱状图
        gender_count.plot(kind='bar', ax=ax1, color=['skyblue', 'dodgerblue'], width=0.8)
        ax1.set_ylabel("人数", fontsize=8)    # Y轴标签
        ax1.set_title("性别分布", fontsize=9) # 图表标题
        ax1.legend(['女', '男'], fontsize=7, loc='upper right')  # 图例
        ax1.tick_params(axis='both', which='major', labelsize=6) # 刻度字体大小
        ax1.grid(axis='y', alpha=0.3)        # Y轴网格线（透明度0.3）
        plt.xticks(rotation=45)              # X轴标签旋转45度
        plt.tight_layout()                   # 自动调整布局
        st.pyplot(fig1)                      # 显示图表
    with col_table:
        # 计算各专业总人数
        total = gender_count.sum(axis=1)
        # 计算性别比例（保留1位小数）
        ratio_df = pd.DataFrame({
            "女 (%)": (gender_count["女"] / total * 100).round(1),
            "男 (%)": (gender_count["男"] / total * 100).round(1)
        })
        st.markdown("##### 性别比例")  # 表格标题
        # 显示表格（设置字体大小9px，保留1位小数）
        st.table(ratio_df.style.format("{:.1f}").set_properties(**{'font-size': '9px'}))

    # 2. 各专业学习指标对比分析
    st.subheader("2. 各专业学习指标对比")
    # 选择要分析的学习指标
    metrics = ["每周学习时长（小时）", "期中考试分数", "期末考试分数"]
    # 按专业分组计算平均值（保留1位小数）
    detail_df = df.groupby("专业")[metrics].mean().round(1)
    # 提取各指标数据
    avg_study = detail_df["每周学习时长（小时）"]
    avg_midterm = detail_df["期中考试分数"]
    avg_final = detail_df["期末考试分数"]
    
    # 布局：图表区(3/4) + 表格区(1/4)
    col_chart, col_table = st.columns([3, 1])
    with col_chart:
        # 创建图表（缩小尺寸：5.5x2.8）
        fig2, ax2 = plt.subplots(figsize=(5.5, 2.8))
        x = np.arange(len(avg_study))  # X轴坐标
        width = 0.35                   # 柱状图宽度
        # 绘制柱状图（学习时长）
        ax2.bar(x, avg_study, width, label='学习时长', alpha=0.8, color='lightblue')
        # 绘制折线图（期中成绩）
        ax2.plot(x, avg_midterm, marker='o', linestyle='--', linewidth=1.2, label='期中', color='orange')
        # 绘制折线图（期末成绩）
        ax2.plot(x, avg_final, marker='s', linestyle='-', linewidth=1.2, label='期末', color='green')
        ax2.set_xlabel('专业', fontsize=8)   # X轴标签
        ax2.set_ylabel('值', fontsize=8)     # Y轴标签
        ax2.set_title('学习指标', fontsize=9)# 图表标题
        ax2.set_xticks(x)                   # 设置X轴刻度
        ax2.set_xticklabels(avg_study.index, rotation=45, fontsize=7)  # X轴标签
        ax2.legend(fontsize=7, loc='upper right')  # 图例
        ax2.grid(axis='y', alpha=0.3)        # Y轴网格线
        plt.tight_layout()                   # 自动调整布局
        st.pyplot(fig2)                      # 显示图表
    with col_table:
        st.markdown("##### 平均值")  # 表格标题
        # 显示表格（设置字体大小9px，保留1位小数）
        st.table(detail_df.style.format("{:.1f}").set_properties(**{'font-size': '9px'}))

    # 3. 各专业出勤率分析
    st.subheader("3. 各专业出勤率分析")
    # 按专业分组计算平均出勤率
    avg_attendance = df.groupby("专业")["上课出勤率"].mean()
    
    # 布局：图表区(3/4) + 表格区(1/4)
    col_chart, col_table = st.columns([3, 1])
    with col_chart:
        # 创建图表（缩小尺寸：5.5x2.8）
        fig3, ax3 = plt.subplots(figsize=(5.5, 2.8))
        # 定义柱状图颜色列表
        colors = ['#FFD700', '#90EE90', '#4169E1', '#FF69B4', '#FFA500', '#87CEEB']
        # 绘制柱状图
        ax3.bar(avg_attendance.index, avg_attendance.values, color=colors[:len(avg_attendance)])
        ax3.set_ylabel('出勤率', fontsize=8)  # Y轴标签
        ax3.set_title('出勤率分布', fontsize=9) # 图表标题
        ax3.set_xticklabels(avg_attendance.index, rotation=45, fontsize=7)  # X轴标签
        ax3.grid(axis='y', alpha=0.3)        # Y轴网格线
        plt.tight_layout()                   # 自动调整布局
        st.pyplot(fig3)                      # 显示图表
    with col_table:
        # 转换为DataFrame并重置索引
        rank_df = avg_attendance.to_frame().reset_index()
        rank_df.columns = ["专业", "出勤率"]  # 重命名列
        st.markdown("##### 排名")  # 表格标题
        # 显示表格（出勤率保留1位百分比，字体大小9px）
        st.table(rank_df.style.format({"出勤率": "{:.1%}"}).set_properties(**{'font-size': '9px'}))

    # 4. 大数据管理专业专项分析
    st.subheader("4. 大数据管理专业专项分析")
    # 筛选大数据管理专业数据
    bigdata_df = df[df["专业"] == "大数据管理"]
    if not bigdata_df.empty:  # 数据非空时展示
        # 关键指标卡片布局（四列）
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            # 平均出勤率（保留1位百分比）
            st.metric("出勤率", f"{bigdata_df['上课出勤率'].mean():.1%}")
        with col2:
            # 平均期末成绩（保留1位小数）
            st.metric("期末成绩", f"{bigdata_df['期末考试分数'].mean():.1f}分")
        with col3:
            # 及格率（保留1位百分比）
            st.metric("及格率", f"{(bigdata_df['期末考试分数'] >= 60).mean():.1%}")
        with col4:
            # 平均学习时长（保留1位小数）
            st.metric("学习时长", f"{bigdata_df['每周学习时长（小时）'].mean():.1f}小时")

        # 成绩分布图表布局（两列）
        col_hist, col_box = st.columns(2)
        with col_hist:
            # 成绩分布直方图（缩小尺寸：5x2.5）
            fig4, ax4 = plt.subplots(figsize=(5, 2.5))
            scores = bigdata_df["期末考试分数"]  # 期末成绩数据
            # 绘制直方图
            ax4.hist(scores, bins=10, edgecolor='black', alpha=0.7, color='green')
            ax4.set_xlabel('期末成绩', fontsize=8)  # X轴标签
            ax4.set_ylabel('频数', fontsize=8)      # Y轴标签
            ax4.set_title('成绩分布', fontsize=9)   # 图表标题
            ax4.tick_params(labelsize=7)           # 刻度字体大小
            ax4.grid(axis='y', alpha=0.3)          # Y轴网格线
            plt.tight_layout()                     # 自动调整布局
            st.pyplot(fig4)                        # 显示图表
        with col_box:
            # 成绩箱线图（交互式）
            fig5 = px.box(bigdata_df, y="期末考试分数", title="成绩箱线图")
            # 调整图表布局
            fig5.update_layout(
                height=250,                       # 图表高度
                margin=dict(t=30, b=10, l=10, r=10),  # 边距
                title_font_size=10,               # 标题字体大小
                font_size=8                       # 整体字体大小
            )
            st.plotly_chart(fig5, use_container_width=True)  # 自适应宽度显示


# 页面3：成绩预测页


else:
    st.title("🔮 期末成绩预测")  # 页面主标题
    # 提示信息
    st.info("请输入学生的学习信息，系统将预测其期末成绩并提供学习建议。")

    # 输入表单布局（两列）
    col1, col2 = st.columns([1, 2])
    with col1:
        # 基础信息输入
        student_id = st.text_input("学号", "2023123456", help="输入学生学号", max_chars=12)
        gender = st.selectbox("性别", ["男", "女"], help="选择性别")
        major = st.selectbox("专业", df["专业"].unique(), help="选择专业")
    with col2:
        # 学习指标输入（滑块）
        study_hours = st.slider("每周学习时长(小时)", 5.0, 30.0, 15.0, 0.5, help="建议每天学习2-3小时")
        attendance = st.slider("上课出勤率", 0.5, 1.0, 0.8, 0.05, help="实际出勤比例")
        midterm_score = st.slider("期中考试分数", 0, 100, 75, help="期中考试成绩")
        homework_rate = st.slider("作业完成率", 0.6, 1.0, 0.9, 0.05, help="作业完成比例")

    # 加载预测模型
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)  # 反序列化加载模型
    except Exception as e:
        # 模型加载失败时显示错误信息并停止程序
        st.error(f"❌ 模型加载失败: {e}")
        st.stop()

    # 预测按钮点击事件
    if st.button("预测期末成绩", type="primary", help="点击预测期末成绩"):
        # 构造模型输入数据（二维数组）
        input_data = np.array([[study_hours, attendance, midterm_score, homework_rate]])
        # 调用模型预测期末成绩
        predicted_score = model.predict(input_data)[0]
        
        st.subheader("📊 预测结果")  # 预测结果标题
        
        # 统一居中容器：三列布局取中间列，实现整体居中
        center_container = st.columns([1, 2, 1])[1]
        with center_container:
            # 限制提示条宽度：单列布局取50%宽度
            score_col = st.columns([0.5])[0]
            with score_col:
                # 根据预测成绩显示不同提示
                if predicted_score >= 60:
                    st.success(f"🎉 预测期末成绩: {predicted_score:.1f} 分")  # 及格：绿色提示
                else:
                    st.error(f"⚠️ 预测期末成绩: {predicted_score:.1f} 分")    # 不及格：红色提示
            
            # 图片居中显示（在同一中间列内）
            st.image(
                # 根据预测结果选择对应图片
                "https://inews.gtimg.com/om_bt/OXIDNDmWuOsJmbMu3_AVgID_o1OYk3-q7EW2d4mnFdr9kAA/641" 
                if predicted_score >= 60 
                else "https://img.soogif.com/sKfXvlCCA8LiMuoXLyZCPT8DEiFI4PIb.jpg",
                # 图片说明文字
                caption="恭喜你！预测结果显示你会及格！" if predicted_score >= 60 else "加油！预测结果显示你需要努力了！",
                width=300  # 图片宽度（可根据需求调整）
            )
