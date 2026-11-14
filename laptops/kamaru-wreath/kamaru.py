"""kamaru.py -- maximality of the free wreath products S_2^+ wr_* S_3^+ and S_3^+ wr_* S_2^+ inside S_6^+,
level by level up to level 6 (levels 3-5 were done in the cloud, redone here as a check).
Result line per level: "C_k = TL_k" means no intermediate quantum group can differ from S_6^+ at level k.
Run:  python3 kamaru.py      Log: kamaru-output/kamaru.log ; numbers: kamaru-output/kamaru.json
Memory: at most about 3 GB (level 6). Stops by itself after 9 hours."""
import json, os, sys, threading, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fcmax
out = os.path.join(HERE, "kamaru-output"); os.makedirs(out, exist_ok=True)
LOG = open(os.path.join(out, "kamaru.log"), "a"); JS = os.path.join(out, "kamaru.json")
try: res = json.load(open(JS))
except Exception: res = {}
T0 = time.time()
def say(s):
    s = "[%s | %5.1f min] %s" % (time.strftime("%H:%M:%S"), (time.time() - T0) / 60, s)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def watchdog():
    time.sleep(9 * 3600); say("9 hours reached, stopping."); os._exit(0)
threading.Thread(target=watchdog, daemon=True).start()
say("start; numpy %s, python %s" % (np.__version__, sys.version.split()[0]))
for n, m, K in [(2, 3, 6), (3, 2, 6)]:
    N = n * m; lab = "S_%d^+ wr S_%d^+" % (n, m); r = res.setdefault(lab, {})
    TL, FC = {}, {}
    for k in range(1, K + 1):
        t0 = time.time()
        TL[k], FC[k], V = fcmax.spaces(n, m, k)
        if k <= 2 or str(k) in r:
            if str(k) in r: say("%s level %d already done: %s" % (lab, k, r[str(k)]["verdict"]))
            continue
        Vlin = fcmax.linear_survivors(V, TL[k - 1], N, k)
        msg = "%s level %d: dim TL %d, FC %d, FC-TL %d, after linear %d" % (lab, k, TL[k].shape[1], FC[k].shape[1], V.shape[1], Vlin.shape[1])
        if Vlin.shape[1]:
            F = fcmax.forms(Vlin, TL, FC, N, k)
            best = fcmax.min_isotropy(F, Vlin.shape[1])
            verdict = "C_k = TL_k" if best > 1e-6 else "INCONCLUSIVE"
            msg += ", %d forms, min = %.3e" % (len(F), best)
        else:
            best, verdict = None, "C_k = TL_k"
        r[str(k)] = {"TL": TL[k].shape[1], "FC": FC[k].shape[1], "linear": int(Vlin.shape[1]), "min": best, "verdict": verdict,
                     "seconds": round(time.time() - t0)}
        json.dump(res, open(JS, "w"), indent=1)
        say(msg + "   => " + verdict + "  (%.0f s)" % (time.time() - t0))
        if verdict != "C_k = TL_k": break
say("all done.")
