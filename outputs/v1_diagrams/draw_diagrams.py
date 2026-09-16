from pathlib import Path
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.font_manager import FontProperties

OUT = Path(__file__).resolve().parent
FONT = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
BOLD = FontProperties(fname='C:/Windows/Fonts/msyhbd.ttc')
plt.rcParams['svg.fonttype'] = 'path'
INK='#183047'; MUTED='#617589'; BLUE='#2878AF'; TEAL='#008779'; PURPLE='#8058B1'; ORANGE='#D76A36'
BG='#FAFCFE'; LINE='#D7E1E9'

def canvas(title, sub):
    fig,ax=plt.subplots(figsize=(18,12))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.set_xlim(0,1800); ax.set_ylim(1200,0); ax.axis('off')
    fig.subplots_adjust(0,0,1,1)
    text(ax,65,62,title,28,bold=True)
    text(ax,65,106,sub,13,color=MUTED)
    return fig,ax

def text(ax,x,y,s,size=14,color=INK,ha='left',bold=False):
    for old,new in [('ₖ₋₁','[k-1]'),('ₖ','[k]'),('₇','7'),('₁','1')]:
        s=s.replace(old,new)
    ax.text(x,y,s,fontsize=size,color=color,ha=ha,va='center',fontproperties=BOLD if bold else FONT,linespacing=1.55)

def box(ax,x,y,w,h,fc='white',ec=LINE,r=15,lw=1.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',facecolor=fc,edgecolor=ec,linewidth=lw))

def arrow(ax,x1,y1,x2,y2,color=INK,lw=2,style='-',rad=0):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=15,color=color,linewidth=lw,linestyle=style,connectionstyle=f'arc3,rad={rad}',shrinkA=2,shrinkB=2))

def route(ax,points,color=INK,lw=2,style='-'):
    xs,ys=zip(*points[:-1]); ax.plot(xs,ys,color=color,lw=lw,ls=style)
    arrow(ax,*points[-2],*points[-1],color,lw,style)

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=150,facecolor=BG)
    fig.savefig(OUT/(name+'.svg'),facecolor=BG)
    plt.close(fig)

def neuron():
    fig,ax=canvas('01  一个 LIF 神经元：历史如何改变当前输出',
        'V1 默认配置 · 固定一个层、一个通道、一个像素位置 · 每个 50 ms 窗口更新一次')
    box(ax,65,147,1670,66,fc='#EEF3FA',ec='#EEF3FA')
    text(ax,90,180,'Δt = 50 ms',16,bold=True)
    text(ax,355,180,'τ 初值 = 200 ms',16,bold=True)
    text(ax,725,180,'β = exp(−Δt / τ) ≈ 0.7788',16,bold=True)
    text(ax,1325,180,'发放阈值 Vth = 1',16,bold=True)

    # Data / synaptic current.
    text(ax,85,255,'① 邻域输入 → 加权电流',17,bold=True,color=BLUE)
    for j,v in enumerate([0.3,1.,0.,0.7]):
        cy=315+j*55
        ax.add_patch(Circle((110,cy),17,fc='#E2EFF8',ec=BLUE,lw=1.5))
        text(ax,110,cy,str(v),10,ha='center',color=BLUE)
        arrow(ax,130,cy,245,398,BLUE,lw=1.2)
    ax.add_patch(Circle((275,398),30,fc='white',ec=BLUE,lw=2))
    text(ax,275,398,'Σ',23,ha='center',color=BLUE)
    text(ax,174,291,'卷积权重 W',12,color=BLUE)
    arrow(ax,307,398,345,398,BLUE)
    box(ax,345,356,185,84,fc='#EAF4FA',ec=BLUE)
    text(ax,437,383,'通道增益 g',16,ha='center',bold=True)
    text(ax,437,416,'Iₖ = g · Conv(input)',12,ha='center')
    text(ax,85,535,'首层：12 通道连续输入',12,color=MUTED)
    text(ax,85,566,'后续层：脉冲或解码器混合特征',12,color=MUTED)
    text(ax,345,479,'g = exp(log_gain)',13,color=BLUE)
    text(ax,345,511,'逐通道可学习；卷积无偏置',11,color=MUTED)

    # Memory is parallel to the current, never fed to input.
    box(ax,600,251,330,77,fc='#F1EBF8',ec=PURPLE)
    text(ax,765,276,'上一窗复位后状态 Uₖ₋₁',15,ha='center',color=PURPLE,bold=True)
    text(ax,765,305,'先乘 β 衰减',13,ha='center',color=PURPLE)
    arrow(ax,765,330,765,365,PURPLE)
    arrow(ax,532,398,594,398,BLUE)
    box(ax,600,366,330,93,fc='white',ec=PURPLE,lw=2)
    text(ax,765,391,'② 累积：复位前膜电位',16,ha='center',bold=True)
    text(ax,765,427,'U_preₖ = β · Uₖ₋₁ + Iₖ',17,ha='center',color=PURPLE)
    text(ax,618,502,'序列首窗：没有历史，U_preₖ = Iₖ',12,color=MUTED)

    arrow(ax,931,398,1012,398,TEAL)
    box(ax,1015,348,305,110,fc='#E7F5F1',ec=TEAL)
    text(ax,1167,374,'③ 阈值发放',17,ha='center',bold=True,color=TEAL)
    text(ax,1167,411,'Sₖ = 1  当 U_preₖ ≥ 1',14,ha='center')
    text(ax,1167,438,'否则 Sₖ = 0',13,ha='center')
    arrow(ax,1322,398,1435,398,TEAL)
    box(ax,1440,354,290,93,fc='#E7F5F1',ec=TEAL)
    text(ax,1585,385,'脉冲 Sₖ → 后续网络层',15,ha='center',bold=True,color=TEAL)
    text(ax,1585,418,'本窗前向输出；每位置至多 1 次',11,ha='center')

    box(ax,1015,523,715,100,fc='#F1EBF8',ec=PURPLE)
    text(ax,1040,551,'④ 软复位并保存',16,bold=True,color=PURPLE)
    text(ax,1040,588,'Uₖ = U_preₖ − 1 × stopgrad(Sₖ)',16,color=PURPLE)
    arrow(ax,1167,460,1167,521,PURPLE)
    route(ax,[(944,410),(960,410),(960,574),(1013,574)],PURPLE,lw=1.5)
    text(ax,1467,552,'→ 下一窗同位置',14,color=PURPLE,bold=True)
    text(ax,1467,590,'不清零；不自动迁移位置',11,color=MUTED)
    text(ax,65,660,'最后一层的特例：分类头读取连续 U_pre₇；S₇ 仅通过复位影响未来状态，不用于当前分类。',13,color=INK)

    # Six-window numeric trajectory.
    box(ax,65,705,1670,317,fc='white')
    text(ax,90,737,'六个窗口的数值示例',18,bold=True)
    text(ax,435,737,'紫色点：复位前 U_pre    空心点：复位后 U    橙色 ↓：发放后减去阈值 1',12,color=MUTED)
    currents=[.45,.50,.60,.30,.75,.20]; pre=[]; post=[]; spikes=[]; u=0; beta=math.exp(-.25)
    for i in currents:
        p=beta*u+i; s=int(p>=1); u=p-s; pre.append(p);post.append(u);spikes.append(s)
    xs=[255+j*265 for j in range(6)]
    y=lambda v:934-v*105
    ax.plot([165,165],[781,939],color=LINE,lw=1.4)
    ax.plot([165,1660],[934,934],color=LINE,lw=1.4)
    ax.plot([165,1660],[y(1),y(1)],color=ORANGE,lw=1.3,ls='--')
    text(ax,145,y(1),'1.0',11,ha='right',color=ORANGE);text(ax,145,y(0),'0',11,ha='right',color=MUTED)
    for j,x in enumerate(xs):
        text(ax,x,776,f'I = {currents[j]:.2f}',12,ha='center',color=BLUE)
        if j:
            ax.plot([xs[j-1],x],[y(post[j-1]),y(pre[j])],color=PURPLE,lw=1.5,alpha=.45)
        ax.scatter([x],[y(pre[j])],s=52,color=PURPLE,zorder=4)
        text(ax,x+15,y(pre[j])-10,f'{pre[j]:.3f}',11,color=PURPLE)
        if spikes[j]:
            arrow(ax,x,y(pre[j])+5,x,y(post[j])-6,ORANGE,lw=2.5)
            ax.scatter([x],[y(post[j])],s=52,facecolors='white',edgecolors=PURPLE,zorder=5)
            text(ax,x+15,y(post[j]),f'{post[j]:.3f}',11,color=MUTED)
        text(ax,x,961,f'窗 {j}',12,ha='center')
        text(ax,x,992,f'S = {spikes[j]}',13,ha='center',color=ORANGE if spikes[j] else MUTED,bold=bool(spikes[j]))
    text(ax,65,1070,'τ = 50 + 1950 · sigmoid(a) ms；每通道一个可学习 a；同通道各位置共享 τ，各自保存 U。',13)
    text(ax,65,1110,'反向传播：脉冲使用代理导数 1 / (1 + |U_pre − 1|)²；仅复位的 S 分支停止梯度。',13)
    text(ax,65,1155,'依据：model/lif2d_stream.py · model/evspsegnet_stream.py｜数值为默认参数下的合成示例',10,color=MUTED)
    save(fig,'01-lif-neuron-dynamics')

def network():
    fig,ax=canvas('02  七组神经元如何组成流式 U-Net',
        '当前窗口在网络中前向流动；每组 LIF 另有一份跨窗口状态 · 圆点仅示意，不表示实际神经元数量')
    box(ax,65,146,1670,81,fc='#EEF3FA',ec='#EEF3FA')
    text(ax,90,173,'事件 (x, y, t, p)',15,bold=True)
    arrow(ax,320,185,381,185,BLUE)
    text(ax,400,173,'50 ms 窗口 → 12 通道计数与归一化',15,bold=True)
    text(ax,400,205,'2 个整窗通道 + 5 × 2 个时间分箱通道',11,color=MUTED)
    arrow(ax,925,185,981,185,BLUE)
    text(ax,1000,173,'Xₖ : [1, 12, 264, 352]',15,color=BLUE,bold=True)
    text(ax,1430,173,'5 个 bin ≠ 5 个时间步',13,color=INK)
    text(ax,1000,205,'每窗只执行一次完整网络前向',11,color=MUTED)

    positions=[(170,333),(350,490),(530,647),(760,802),(995,647),(1200,490),(1410,333)]
    names=['enc1','enc2','enc3','enc4','dec3','dec2','dec1']
    channels=['12 → 12','12 → 24','24 → 48','48 → 48','96 → 48','48 → 24','24 → 12']
    shapes=['12 × 264 × 352','24 × 132 × 176','48 × 66 × 88','48 × 33 × 44','48 × 66 × 88','24 × 132 × 176','12 × 264 × 352']
    w,h=170,124

    # Main flow connections, drawn behind units.
    for j in range(3):
        x,y=positions[j];nx,ny=positions[j+1]
        arrow(ax,x+w,y+h-10,nx+15,ny-4,TEAL,lw=2.5)
        text(ax,x+w+15,y+h+31,'S'+str(j+1),12,color=TEAL)
    for a,b,label in [(3,4,'up3'),(4,5,'up2'),(5,6,'up1')]:
        x,y=positions[a];nx,ny=positions[b]
        bx,by=nx-205,ny+35
        arrow(ax,x+85,y-2,bx+65,by+56,TEAL,lw=2.3)
        box(ax,bx,by,130,54,fc='#FFF1E7',ec=ORANGE,r=9)
        text(ax,bx+65,by+16,label,12,ha='center',color=ORANGE,bold=True)
        text(ax,bx+65,by+39,'ConvT 2×2 / s2',10,ha='center',color=ORANGE)
        arrow(ax,bx+132,by+27,nx-40,ny+62,BLUE,lw=2.2)
    # Skips, with explicit concatenation nodes.
    for a,b in [(0,6),(1,5),(2,4)]:
        x,y=positions[a];nx,ny=positions[b]
        yy=y-43; joinx=nx-24
        route(ax,[(x+85,y),(x+85,yy),(joinx,yy),(joinx,ny+61)],TEAL,1.8,'--')
        ax.add_patch(Circle((joinx,ny+62),14,fc='white',ec=TEAL,lw=1.5))
        text(ax,joinx,ny+62,'C',11,ha='center',color=TEAL,bold=True)
        arrow(ax,joinx+15,ny+62,nx-1,ny+62,TEAL,lw=1.6)
        text(ax,(x+85+joinx)/2,yy-17,'S'+str(a+1)+'  脉冲跳连 → 拼接',12,ha='center',color=TEAL)

    for j,((x,y),name,ch,sh) in enumerate(zip(positions,names,channels,shapes)):
        box(ax,x,y,w,h,fc='white',ec=BLUE if j<4 else TEAL,lw=1.8)
        text(ax,x+14,y+23,name,16,bold=True)
        text(ax,x+w-12,y+23,'LIF '+str(j+1),11,ha='right',color=PURPLE)
        text(ax,x+14,y+50,ch+' · Conv 3×3',10,color=MUTED)
        # Representative neuron population, mixed active/inactive units.
        for r in range(2):
            for c in range(5):
                active=(r+c+j)%3==0
                ax.add_patch(Circle((x+24+c*28,y+77+r*23),6.5,fc=TEAL if active else '#E9F0F3',ec=TEAL if active else '#C0CED8',lw=.9))
        text(ax,x+85,y+h+21,sh,10,ha='center',color=MUTED)
        # Local recurrent loop, not a cross-layer state bus.
        route(ax,[(x+164,y+103),(x+188,y+103),(x+188,y+8),(x+164,y+8)],PURPLE,1.4)
        text(ax,x+196,y+59,'U'+str(j+1),10,color=PURPLE)

    route(ax,[(1270,227),(1270,248),(95,248),(95,395),(167,395)],BLUE,lw=2)
    text(ax,75,421,'Xₖ',13,color=BLUE)
    # Last layer analogue readout.
    route(ax,[(1582,395),(1686,395),(1686,535)],BLUE,lw=2.2)
    text(ax,1632,366,'U_pre₇',12,ha='center',color=BLUE)
    box(ax,1468,537,266,185,fc='#EAF4FA',ec=BLUE)
    text(ax,1601,566,'逐事件连续读出',16,ha='center',color=BLUE,bold=True)
    text(ax,1601,604,'按 (y, x) 取 12 维 U_pre₇',11,ha='center')
    text(ax,1601,634,'拼接 p、t_local → 14 维',11,ha='center')
    text(ax,1601,669,'MLP 14 → 32 → 1',14,ha='center',bold=True)
    text(ax,1601,698,'隐藏层 ReLU；输出 logit',11,ha='center',color=MUTED)
    arrow(ax,1601,724,1601,758,BLUE)
    box(ax,1468,761,266,81,fc='white',ec=BLUE)
    text(ax,1601,789,'sigmoid → 每个事件的概率',12,ha='center',bold=True)
    text(ax,1601,819,'按原始下标回填',11,ha='center',color=MUTED)

    # Unit key with actual neuron population meaning.
    box(ax,65,975,1670,168,fc='white')
    text(ax,90,1008,'一个方块代表一组神经元',16,bold=True)
    text(ax,90,1045,'Conv 3×3 → 通道增益 g → LIF',13)
    text(ax,90,1081,'每个 [通道, y, x] 有独立膜电位',12,color=MUTED)
    text(ax,90,1113,'同通道共享 τ；默认不用 BN / GN',12,color=MUTED)
    text(ax,675,1008,'跨窗状态：七份各自循环',16,bold=True,color=PURPLE)
    text(ax,675,1045,'U₁…U₇ 从本窗传到下一窗对应层',12)
    text(ax,675,1081,'carry 保留；reset 每窗忽略历史',12,color=MUTED)
    text(ax,675,1113,'不同 NPZ 序列之间清空状态',12,color=MUTED)
    text(ax,1250,1008,'路径图例',16,bold=True)
    text(ax,1250,1045,'绿色：脉冲 S / 脉冲输入路径',12,color=TEAL)
    text(ax,1250,1078,'蓝色：连续值    紫色：状态 U',12,color=BLUE)
    text(ax,1250,1111,'橙色 ConvT 输出连续值；C = 拼接',11,color=ORANGE)
    text(ax,65,1171,'默认网络：105,345 参数 · 3 次空间下采样 · 普通稠密 Conv2d / ConvTranspose2d · 依据 model/evspsegnet_stream.py',10,color=MUTED)
    save(fig,'02-network-neuron-flow')

if __name__=='__main__':
    neuron();network()
    print('Created two PNG figures and two SVG vector figures in',OUT)
