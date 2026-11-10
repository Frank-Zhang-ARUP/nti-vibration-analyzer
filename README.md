# NTi WAV Vibration Analyzer

## 功能
- 多 WAV、多通道上传
- 电压按传感器灵敏度换算为加速度
- 时域加速度与指数 RMS 加速度级
- Hann 窗单边 Peak 幅值谱
- Welch PSD 峰值识别，输出楼板主导自振频率候选
- 峰值加速度、RMS、峰值因数
- 分析结果导出 Excel

## 本地运行
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 在线部署
将 `app.py` 与 `requirements.txt` 上传到 GitHub 仓库，再在 Streamlit Community Cloud 中选择该仓库和 `app.py` 部署。

## 标定要点
1. 默认传感器灵敏度为 10.2 mV/(m/s²)，取自本批 NTi CS011 测试配置。
2. 必须确认采集链路增益及 WAV 的 0 dBFS 对应输入峰值电压。
3. 若 WAV 已储存工程单位 m/s²，可勾选“WAV 样值已直接表示 m/s²”。
4. 未完成幅值标定时，频率结果仍可用于初步判断，但峰值加速度只能视为相对值。

## 自振频率判断限制
程序将搜索频段内 Welch PSD 的显著峰作为候选。稳定设备转频、步频及其谐波可能被误判为结构模态。建议用锤击、自由衰减、设备启停对比或多测点一致性进行确认。
