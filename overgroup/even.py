"""Further checks of the zigzag inequality at even levels (Theorem 3.5 proves it at all even levels):
complete enumeration of the identity words at level 12, and of the identity words with a negative window at levels 14 to 20.
Output: even.log"""
import time, overgroup as og
from jw import nu_words
log = open("even.log", "w")
def say(s): print(s, flush=True); log.write(s + "\n"); log.flush()
for k, full in [(14, False), (16, False), (18, False), (12, True), (20, False)]:
    t0 = time.time(); L, s = og.zigzag_table(k); nu = nu_words(L, list(s)); mu = sum(nu[w] * v for w, v in s.items())
    n, m = og.min_score(k, L, s, full=full)
    say("level %d: mu = %s, k mu = %.3e; %s: %d, minimal score %d  (%.0fs)" % (k, mu, float(k * mu),
        "all identity words" if full else "identity words with a negative window", n, m, time.time() - t0))
