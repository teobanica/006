import sys, json, fcplanar
n, m, K = map(int, sys.argv[1:4])
f = open("log-%d-%d.txt" % (n, m), "w")
def log(s): print(s); f.write(s + "\n"); f.flush()
out = fcplanar.run(n, m, K, log=log)
json.dump({str(k): v for k, v in out.items()}, open("fcplanar-%d-%d.json" % (n, m), "w"), indent=1)
