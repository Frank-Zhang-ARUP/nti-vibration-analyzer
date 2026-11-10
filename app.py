import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy.io import wavfile
from scipy import signal
import plotly.graph_objects as go

st.set_page_config(page_title="NTi WAV 振动分析", layout="wide")
st.title("NTi WAV 振动分析")
st.caption("上传 WAV，生成时域加速度级、Peak 幅值谱，并估算楼板主导自振频率与峰值加速度。")

with st.sidebar:
    st.header("分析设置")
    sensitivity_mv = st.number_input("传感器灵敏度 [mV/(m/s²)]", 0.001, 10000.0, 10.2, 0.1)
    gain = st.number_input("前置放大/记录链路增益", 0.000001, 1000000.0, 1.0, 0.1)
    wav_full_scale_v = st.number_input("WAV 满刻度峰值对应电压 [Vpeak]", 0.000001, 1000.0, 1.0, 0.1,
        help="若 WAV 已是工程单位加速度，请勾选下面选项。若是数字满刻度，必须填写记录仪 0 dBFS 对应的输入峰值电压。")
    engineering_units = st.checkbox("WAV 样值已直接表示 m/s²", False)
    ref_acc = st.number_input("加速度级参考值 [m/s²]", format="%.2e", value=1e-6)
    tau = st.number_input("时域加速度级 RMS 时间常数 [s]", 0.01, 20.0, 1.0, 0.1)
    fmin = st.number_input("自振频率搜索下限 [Hz]", 0.1, 2000.0, 3.0, 0.5)
    fmax = st.number_input("自振频率搜索上限 [Hz]", 0.2, 5000.0, 80.0, 1.0)
    prominence_db = st.number_input("候选峰最小突出度 [dB]", 1.0, 40.0, 6.0, 1.0)
    highpass = st.number_input("高通截止频率 [Hz]（0=关闭）", 0.0, 1000.0, 0.5, 0.1)

files = st.file_uploader("上传一个或多个 WAV 文件", type=["wav"], accept_multiple_files=True)


def decode_wav(raw):
    fs, x = wavfile.read(io.BytesIO(raw))
    if x.ndim == 1: x = x[:, None]
    if np.issubdtype(x.dtype, np.integer):
        info=np.iinfo(x.dtype); denom=max(abs(info.min),info.max)
        xn=x.astype(np.float64)/denom
    else:
        xn=x.astype(np.float64)
    return fs, xn

def to_acc(xn):
    if engineering_units: return xn
    volts=xn*wav_full_scale_v
    return volts/(sensitivity_mv/1000.0*gain)

def hp_filter(a, fs):
    if highpass <= 0 or highpass >= fs/2: return a
    sos=signal.butter(4, highpass, btype='highpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos,a)

def exp_rms_level(a, fs):
    alpha=np.exp(-1/(max(tau,1/fs)*fs))
    ms=signal.lfilter([1-alpha],[1,-alpha],a*a)
    rms=np.sqrt(np.maximum(ms, np.finfo(float).tiny))
    return 20*np.log10(rms/ref_acc)

def peak_spectrum(a, fs):
    n=len(a); w=signal.windows.hann(n, sym=False); cg=w.mean()
    y=np.fft.rfft((a-np.mean(a))*w)
    amp=np.abs(y)/(n*cg)
    if n>2: amp[1:-1]*=2
    f=np.fft.rfftfreq(n,1/fs)
    return f,amp

def find_modes(a, fs):
    nper=min(len(a), max(1024, int(fs*8)))
    f, p=signal.welch(a,fs=fs,window='hann',nperseg=nper,noverlap=nper//2,detrend='constant',scaling='density')
    db=10*np.log10(np.maximum(p,np.finfo(float).tiny))
    mask=(f>=fmin)&(f<=min(fmax,fs/2))
    fm,dm=f[mask],db[mask]
    if len(fm)<3: return f,db,[],None
    idx,props=signal.find_peaks(dm,prominence=prominence_db,distance=max(1,int(0.5/(fm[1]-fm[0]))))
    candidates=[]
    for k,prom in zip(idx,props.get('prominences',[])):
        candidates.append((float(fm[k]),float(dm[k]),float(prom)))
    candidates=sorted(candidates,key=lambda z:z[1],reverse=True)
    dominant=candidates[0][0] if candidates else float(fm[np.argmax(dm)])
    return f,db,candidates[:8],dominant

if files:
    summaries=[]; exports={}
    for uf in files:
        raw=uf.getvalue(); fs,xn=decode_wav(raw)
        for ch in range(xn.shape[1]):
            label=f"{uf.name} | Ch {ch+1}"
            a=hp_filter(to_acc(xn[:,ch]),fs)
            t=np.arange(len(a))/fs
            lvl=exp_rms_level(a,fs)
            f,amp=peak_spectrum(a,fs)
            fw,pdb,cands,fn=find_modes(a,fs)
            peak=float(np.max(np.abs(a))); rms=float(np.sqrt(np.mean(a*a)))
            crest=peak/rms if rms>0 else np.nan
            summaries.append({"文件/通道":label,"采样率_Hz":fs,"时长_s":len(a)/fs,"峰值加速度_m_s2":peak,
                              "RMS加速度_m_s2":rms,"峰值因数":crest,"主导候选频率_Hz":fn})
            tab1,tab2,tab3=st.tabs([f"{label} 时域", "Peak 幅值谱", "自振频率"])
            with tab1:
                c1,c2,c3,c4=st.columns(4)
                c1.metric("峰值加速度",f"{peak:.5g} m/s²")
                c2.metric("RMS 加速度",f"{rms:.5g} m/s²")
                c3.metric("峰值因数",f"{crest:.2f}")
                c4.metric("主导候选频率",f"{fn:.2f} Hz" if fn else "未识别")
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=t,y=a,name="Acceleration",line=dict(width=1)))
                fig.update_layout(xaxis_title="Time [s]",yaxis_title="Acceleration [m/s²]",height=380)
                st.plotly_chart(fig,use_container_width=True)
                fig2=go.Figure(go.Scatter(x=t,y=lvl,name="Acceleration level",line=dict(width=1)))
                fig2.update_layout(xaxis_title="Time [s]",yaxis_title=f"Acceleration level [dB re {ref_acc:g} m/s²]",height=380)
                st.plotly_chart(fig2,use_container_width=True)
            with tab2:
                mask=(f>0)&(f<=min(fmax*3,fs/2))
                fig=go.Figure(go.Scatter(x=f[mask],y=amp[mask],line=dict(width=1)))
                fig.update_layout(xaxis_title="Frequency [Hz]",yaxis_title="Peak amplitude [m/s²]",height=430)
                st.plotly_chart(fig,use_container_width=True)
            with tab3:
                mask=(fw>=fmin)&(fw<=min(fmax,fs/2))
                fig=go.Figure(go.Scatter(x=fw[mask],y=pdb[mask],name="Welch PSD"))
                if cands:
                    fig.add_trace(go.Scatter(x=[c[0] for c in cands],y=[c[1] for c in cands],mode='markers',name='Candidate peaks'))
                fig.update_layout(xaxis_title="Frequency [Hz]",yaxis_title="PSD [dB (m/s²)²/Hz]",height=430)
                st.plotly_chart(fig,use_container_width=True)
                st.dataframe(pd.DataFrame(cands,columns=["频率_Hz","PSD_dB","突出度_dB"]),use_container_width=True)
                st.warning("主导谱峰仅作为楼板自振频率候选。若记录中含稳定机械转频、步频或冲击重复频率，应通过停机对比、锤击/自由衰减测试或多测点模态一致性排除强迫频率。")
            step=max(1,int(fs/100))
            exports[label]=pd.DataFrame({"Time_s":t[::step],"Acceleration_m_s2":a[::step],"AccelerationLevel_dB":lvl[::step]})
    st.subheader("汇总")
    summary_df=pd.DataFrame(summaries)
    st.dataframe(summary_df,use_container_width=True)
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine='openpyxl') as writer:
        summary_df.to_excel(writer,index=False,sheet_name='Summary')
        for i,(name,df) in enumerate(exports.items(),1): df.to_excel(writer,index=False,sheet_name=f'Time_{i:02d}')
    st.download_button("下载分析结果 Excel",buf.getvalue(),"vibration_analysis.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("请上传 WAV 文件。默认按 NTi CS011 灵敏度 10.2 mV/(m/s²) 设置，但仍需确认 WAV 满刻度对应电压和记录链路增益，否则绝对加速度与峰值加速度可能不正确。")
