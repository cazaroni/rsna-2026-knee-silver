"""Per-study knee side for training: DICOM Laterality tag first, report text second.

Writes laterality.csv (StudyInstanceUID, tag, report, side, source) and prints how
well the report matcher agrees with the tag where both exist.
"""
import json, re, collections
import pandas as pd

KNEE = r"(knee|rodilla|diz|knie|kniegelenk|γονατ|γόνατ|колян|колен|koljen|genou|ginocchio|joelho)"
RIGHT = (r"right|rt\.?|derech[ao]|der\.?|sa[gğ]|rechts?|rechter|rechten|rechtes|recht"
         r"|δεξι|∆εξι|δεξ|дясн|десн|desn\w*|droite?|destr[oa]|direit[oa]")
LEFT = (r"left|lt\.?|izquierd[ao]|izq\.?|sol|links?|linker|linken|linkes"
        r"|αριστερ|ляв|лев|lijev\w*|gauche|sinistr[oa]|esquerd[oa]")


def side_from_report(text):
    s = str(text).lower().replace("∆", "δ")
    near = lambda side: len(re.findall(rf"\b(?:{side})\w*\W+(?:\w+\W+)?{KNEE}", s)) + \
        len(re.findall(rf"{KNEE}\w*\W+(?:\w+\W+)?(?:{side})\b", s))
    r, l = near(RIGHT), near(LEFT)
    if r and not l:
        return "R"
    if l and not r:
        return "L"
    if r and l:
        return "both"
    return ""


def norm_tag(tags):
    t = {x[:1] for x in tags if x[:1] in "LR"}
    return t.pop() if len(t) == 1 else ("both" if len(t) > 1 else "")


meta = [json.loads(l) for l in open("meta2.jsonl")]
tags = collections.defaultdict(list)
for m in meta:
    tags[m["study"]] += m.get("laterality_tag", [])

tr = pd.read_csv("train.csv")
rows = []
for u, rep in zip(tr.StudyInstanceUID, tr.Report):
    tg, rp = norm_tag(tags.get(u, [])), side_from_report(rep)
    if tg in ("L", "R"):
        side, src = tg, "tag"
    elif rp in ("L", "R"):
        side, src = rp, "report"
    else:
        side, src = "", "unknown"
    rows.append((u, tg, rp, side, src))
df = pd.DataFrame(rows, columns=["StudyInstanceUID", "tag", "report", "side", "source"])
df.to_csv("laterality.csv", index=False)

print(df.source.value_counts().to_string())
both = df[df.tag.isin(["L", "R"]) & df.report.isin(["L", "R"])]
print(f"report vs tag where both known: {len(both)} studies, agree {(both.tag == both.report).mean():.3f}")
print("tag conflicts within study:", (df.tag == "both").sum(), "| report says both:", (df.report == "both").sum())
print("side counts:", df.side.value_counts().to_dict())
