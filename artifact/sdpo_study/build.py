import json, re
src = open("/home/soumojib/.claude/jobs/707e3708/tmp/sdpo_page/template.html").read()
data = json.load(open("/home/soumojib/repos/toypref-robo/results/robol/sdpo_study.json"))
html = src.replace("__DATA__", json.dumps(data, separators=(",", ":")))
open("/home/soumojib/.claude/jobs/707e3708/tmp/sdpo_page/index.html", "w").write(html)
js = html[html.index("<script>") + 8: html.rindex("</script>")]
open("/home/soumojib/.claude/jobs/707e3708/tmp/sdpo_page/chk.js", "w").write(js)
# crude balance check (no node on this host)
pairs = {"(": ")", "[": "]", "{": "}"}
stack, in_str, q, esc, tpl = [], False, "", False, 0
for ch in js:
    if in_str:
        if esc: esc = False
        elif ch == "\\": esc = True
        elif ch == q: in_str = False
        continue
    if ch in "\"'`": in_str, q = True, ch; continue
    if ch in pairs: stack.append(pairs[ch])
    elif ch in ")]}":
        if not stack or stack.pop() != ch: print("UNBALANCED at", ch); break
print("size", len(html), "stack left", len(stack))
print("keys", list(data.keys()))
print("drift arms", list(data["drift"].keys()))
print("variants", {k: list(v.keys()) for k, v in data["variants_best"].items()})
