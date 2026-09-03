# Round-2 charts + windowed stats; reuses the round-1 script's style/logic.
import json, statistics, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid"); sns.set_context("notebook")
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":"dimgrey","axes.labelcolor":"dimgrey",
    "axes.titlecolor":"0.2","xtick.color":"dimgrey","ytick.color":"dimgrey","legend.frameon":False,
    "figure.facecolor":"white","savefig.facecolor":"white"})
DARK2 = sns.color_palette("Dark2"); COL = {"close": DARK2[0], "noclose": DARK2[1]}
POOL = 2_340_338
OUT = Path("/Users/bobzetian/projects/sessioncontrol/llm-d/guides/session-control/benchmark/close-session-ab-20260902")

def load(arm):
    rows=[]; data=json.load(open(f"/tmp/close-session-bench/ab-{arm}/reports/per_request_lifecycle_metrics.json"))
    t0=min(e["start_time"] for e in data)
    for e in data:
        if e.get("error"): continue
        info=e["info"]; rm=info["response_metrics"]; su=rm.get("server_usage") or {}
        det=su.get("prompt_tokens_details") or {}; cm=e.get("computed_metrics") or {}
        ttft=cm.get("time_to_first_token")
        if ttft is None and rm.get("chunk_times"): ttft=rm["chunk_times"][0]-e["start_time"]
        rows.append(dict(t=(e["start_time"]-t0)/60.0, sub="_sa_" in info["graph_event_id"],
                         gid=info["graph_event_id"], ttft=ttft,
                         prompt=su.get("prompt_tokens"), cached=det.get("cached_tokens")))
    rows.sort(key=lambda r:r["t"]); return rows

arms={a:load(a) for a in ("close","noclose")}
hit=lambda r:(r["cached"]/r["prompt"]) if (r["prompt"] and r["cached"] is not None) else None
def pct(v,p): v=sorted(v); return v[min(len(v)-1,int(len(v)*p/100))] if v else float("nan")

fill={}
for arm,rows in arms.items():
    cum,tf=0,None
    for r in rows:
        if r["prompt"] is not None and r["cached"] is not None:
            cum+=max(0,r["prompt"]-r["cached"])
            if cum>=POOL and tf is None: tf=r["t"]
    fill[arm]=tf

stats={}
for arm,rows in arms.items():
    par=[r for r in rows if not r["sub"]]; fl=fill[arm] or 10
    late=[r for r in par if r["t"]>=fl]
    def blk(rs):
        tt=[r["ttft"] for r in rs if r["ttft"] is not None]
        hh=[hit(r) for r in rs if hit(r) is not None]
        losses=[r for r in rs if hit(r) is not None and hit(r)<0.5 and not r["gid"].endswith("turn_0")]
        return dict(n=len(rs),p50=pct(tt,50),p90=pct(tt,90),p99=pct(tt,99),
                    hit=statistics.mean(hh) if hh else None,losses=len(losses))
    cum=sum(max(0,r["prompt"]-r["cached"]) for r in rows if r["prompt"] is not None and r["cached"] is not None)
    stats[arm]=dict(parent=blk(par),late=blk(late),fill=fl,written_M=cum/1e6)
print(json.dumps(stats,indent=1,default=float))

def rolling(rows,key,w=51):
    pts=[(r["t"],key(r)) for r in rows if key(r) is not None]
    return ([p[0] for p in pts],
            [statistics.median(q[1] for q in pts[max(0,i-w//2):i+w//2+1]) for i in range(len(pts))])

# timeline r2
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,6.6),sharex=True)
fig.subplots_adjust(left=0.08,right=0.97,top=0.90,bottom=0.09,hspace=0.15)
for arm in arms:
    par=[r for r in arms[arm] if not r["sub"]]
    x,y=rolling(par,lambda r:r["ttft"]); ax1.plot(x,y,color=COL[arm],lw=1.8,label=arm)
    x,y=rolling(par,hit); ax2.plot(x,y,color=COL[arm],lw=1.8)
    if fill[arm]:
        for ax in (ax1,ax2): ax.axvline(fill[arm],color=COL[arm],lw=1,ls=(0,(4,3)),alpha=0.7)
ax1.set_ylabel("rolling median TTFT (s)"); ax1.legend(loc="upper left")
ax1.set_title("Main-agent turns; dashed = est. pool fill", loc="left")
ax2.set_ylabel("rolling median cache-hit ratio"); ax2.set_xlabel("minutes since arm start"); ax2.set_ylim(0,1.02)
for ax in (ax1,ax2): sns.despine(ax=ax,left=True,bottom=True)
fig.suptitle("Round 2 (idle cap 30s, 16 sessions, 80 min): arms remain superimposed",
             x=0.08,ha="left",fontsize=13,fontweight="bold",color="0.2")
fig.savefig(OUT/"timeline-r2.png",dpi=180); plt.close(fig)

# prefix-loss r2
fig,ax=plt.subplots(figsize=(9,4.0)); fig.subplots_adjust(left=0.08,right=0.97,top=0.86,bottom=0.14)
buckets=np.arange(0,90,10); width=3.6
for k,arm in enumerate(arms):
    par=[r for r in arms[arm] if not r["sub"]]
    losses=[r["t"] for r in par if hit(r) is not None and hit(r)<0.5 and not r["gid"].endswith("turn_0")]
    counts,_=np.histogram(losses,bins=list(buckets)+[90])
    ax.bar(buckets+1.2+k*width,counts,width=width,align="edge",color=COL[arm],label=arm,
           edgecolor="white",linewidth=0.5)
ax.set_title("Round 2: prefix-loss events per 10-minute bucket", loc="left")
ax.set_xlabel("minutes since arm start"); ax.set_ylabel("events"); ax.set_xticks(buckets); ax.legend()
sns.despine(ax=ax,left=True,bottom=True)
fig.savefig(OUT/"prefix-loss-r2.png",dpi=180); plt.close(fig)

# cross-round comparison bars
r1={"close":{"p90":1.30,"hit":0.865,"losses":72,"late_hit":0.891},
    "noclose":{"p90":1.32,"hit":0.866,"losses":73,"late_hit":0.891}}
fig,axes=plt.subplots(1,3,figsize=(11,3.8)); fig.subplots_adjust(wspace=0.3,left=0.06,right=0.98,top=0.80,bottom=0.18)
metrics=[("TTFT p90 (s)","p90",lambda a,arm: r1[arm]["p90"] if a==1 else stats[arm]["parent"]["p90"]),
         ("cache-hit mean","hit",lambda a,arm: r1[arm]["hit"] if a==1 else stats[arm]["parent"]["hit"]),
         ("prefix-loss events","losses",lambda a,arm: r1[arm]["losses"] if a==1 else stats[arm]["parent"]["losses"])]
xt=[0,1]; labels=["round 1\n(idle 1s)","round 2\n(idle 30s)"]
for ax,(title,_,get) in zip(axes,metrics):
    for k,arm in enumerate(("close","noclose")):
        vals=[get(1,arm),get(2,arm)]
        ax.bar([x+k*0.35 for x in xt],vals,width=0.35,color=COL[arm],label=arm,
               edgecolor="white",linewidth=0.5)
        for x,v in zip(xt,vals):
            ax.annotate(f"{v:.2f}" if v<10 else f"{v:.0f}",(x+k*0.35,v),ha="center",va="bottom",fontsize=9,color="dimgrey")
    ax.set_xticks([x+0.175 for x in xt]); ax.set_xticklabels(labels,fontsize=9.5)
    ax.set_title(title,loc="left",fontsize=11)
    sns.despine(ax=ax,left=True,bottom=True)
axes[0].legend(fontsize=9)
fig.suptitle("Main-agent metrics across rounds: pressure rose (round 2), close-vs-noclose gap never appeared",
             x=0.06,ha="left",fontsize=12.5,fontweight="bold",color="0.2")
fig.savefig(OUT/"rounds-compare.png",dpi=180); plt.close(fig)
print("charts done")
