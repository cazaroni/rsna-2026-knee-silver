"""Relabel knee MRI reports against the hosts' expert criteria (forum topic 733343) with Claude.

usage:
  relabel.py pilot <model> <n> <out.jsonl>        direct calls on n non-gold training reports (fixed sample), 8 threads
  relabel.py batch <model> <out_batch_id.txt>     submit every training report as one Message Batch (50% price)
  relabel.py collect <batch_id> <out.jsonl>       download a finished batch
  relabel.py gold <model> <out.jsonl>             direct calls on the 58 gold reports (label-source check only)

Per finding the model returns: status (explicit_positive / explicit_negative / uncertain / not_mentioned),
p = probability that expert radiologists reading the images would mark the finding positive under the hosts'
criterion, and a short evidence quote. The API key is read from ~/.anthropic_key and never printed.
"""
import json, os, sys, random, time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

KEYS = {"ACL": "ACL", "MCL": "MCL", "MedialMeniscus": "Medial Meniscus", "LateralMeniscus": "Lateral Meniscus",
        "MedialOA": "Medial OA", "LateralOA": "Lateral OA", "PFOA": "PF OA", "Effusion": "Effusion",
        "Synovitis": "Synovitis", "Bakers": "Baker's", "Contusion": "Contusion", "Fracture": "Fracture"}

SYSTEM = """You convert one knee MRI radiology report (any language) into labels for a machine-learning competition. The labels must predict what two expert musculoskeletal radiologists decided when they later read the IMAGES of the same examination, using the criteria below. The report is the only evidence you have; the experts never saw it.

EXPERT CRITERIA (verbatim from the competition hosts). Borderline or ambiguous findings were graded NEGATIVE to favour specificity. Labels are for a single knee, the examined one.
- ACL: a high-grade partial or full-thickness tear of the anterior cruciate ligament, meaning complete discontinuity of the ligament, or more than 50 percent of fibers disrupted, with or without secondary signs such as characteristic pivot-shift bone contusions. Mild signal change, degeneration, or thickening without discontinuity is graded negative.
- MCL: a high-grade partial or complete acute tear of the medial collateral ligament, with disrupted fibers and edema within and adjacent to the ligament. Low-grade sprains and chronic or remote stress changes are graded negative.
- MedialMeniscus: abnormal signal that definitely contacts the meniscal surface on at least two images, or a morphologic abnormality such as a truncated, diminutive, or displaced fragment, involving the medial meniscus. Intrasubstance degeneration that does not reach the surface is negative.
- LateralMeniscus: the same criteria applied to the lateral meniscus.
- MedialOA: a moderate or large area (roughly 1 cm or greater) of high-grade cartilage loss, defined as greater than 50 percent of cartilage thickness, in the medial (femorotibial) compartment, with or without underlying subchondral marrow changes.
- LateralOA: the same criteria applied to the lateral (femorotibial) compartment.
- PFOA: the same criteria applied to the patellofemoral compartment (patella and trochlea).
- Effusion: a moderate or large amount of fluid distending the joint.
- Synovitis: inflammation and thickening of the synovial lining of the joint.
- Bakers: a moderate or large fluid collection in the characteristic location behind the knee (Baker / popliteal cyst).
- Contusion: a bone contusion, seen as bone marrow edema-like signal from impact, without a discrete fracture line.
- Fracture: an acute cortical break or fracture line.

FOR EACH OF THE 12 FINDINGS RETURN
1. status - what the report says about this finding in the examined knee now:
   "explicit_positive": the report states the finding is present (any grade or size);
   "explicit_negative": the report states it is absent or the structure is normal/intact;
   "uncertain": hedged, possible, suspected, cannot exclude, or partly described;
   "not_mentioned": the report says nothing about this finding.
   Ignore the clinical question or indication ("rule out tear"), prior history and old surgery unless the report describes the current state, and anything about the other knee.
2. p - your probability, from 0 to 1, that the experts marked this finding POSITIVE under its criterion. Be calibrated:
   - Use the thresholds above. A finding the report describes as below the criterion (small or trace effusion, low-grade or grade I sprain, intrasubstance or degenerative meniscal signal, mild chondropathy, superficial or focal cartilage change, small Baker cyst, degenerative or subchondral edema called osteoarthritis rather than a bruise) should get a LOW p, not a medium one.
   - A clear statement that meets the criterion should get a high p (about 0.85 to 0.97); explicit normality a very low p (about 0.02 to 0.05).
   - Hedged wording sits in between and below 0.5 unless the description itself meets the criterion.
   - When the report does not mention the finding, radiologists often simply did not comment. Estimate p from the rest of the report (for example a large effusion or inflammatory arthropathy raises Synovitis; a pivot-shift pattern of bone bruises raises ACL; a fracture line description is Fracture, not Contusion), otherwise use a low base-rate probability.
3. evidence - the shortest quote from the report (original language, at most 120 characters) that drove your answer, or "" if not mentioned.
Answer with the JSON object only."""

FIND = {"type": "object", "additionalProperties": False, "required": ["status", "p", "evidence"],
        "properties": {"status": {"type": "string", "enum": ["explicit_positive", "explicit_negative", "uncertain", "not_mentioned"]},
                       "p": {"type": "number"}, "evidence": {"type": "string"}}}
SCHEMA = {"type": "object", "additionalProperties": False, "required": list(KEYS), "properties": {k: FIND for k in KEYS}}

def params(model, report):
    return dict(model=model, max_tokens=8000,
                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": f"Report:\n{report}"}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}})

def client():
    return anthropic.Anthropic(api_key=open(os.path.expanduser("~/.anthropic_key")).read().strip(), max_retries=4)

def reports(gold_only=False):
    t = pd.read_csv("train.csv").set_index("StudyInstanceUID")
    gold = set(t.index[t[list(KEYS.values())].notna().any(axis=1)])
    r = t.Report.dropna()
    return r[r.index.isin(gold)] if gold_only else r[~r.index.isin(gold)]

def parse(msg):
    text = next(b.text for b in msg.content if b.type == "text")
    return json.loads(text)

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode in ("pilot", "gold"):
        if mode == "gold":
            model, out = sys.argv[2], sys.argv[3]; R = reports(gold_only=True); ids = sorted(R.index); n = len(ids)
        else:
            model, n, out = sys.argv[2], int(sys.argv[3]), sys.argv[4]
            R = reports(); ids = sorted(R.index); random.Random(20261012).shuffle(ids); ids = ids[:n]
        c = client(); usage = {"in": 0, "cache_r": 0, "cache_w": 0, "out": 0}
        def one(u):
            for attempt in range(3):
                try:
                    m = c.messages.create(**params(model, R[u]))
                    if m.stop_reason != "end_turn": return u, None, f"stop_reason {m.stop_reason}"
                    return u, (parse(m), m.usage), None
                except (json.JSONDecodeError, StopIteration) as e:
                    err = repr(e)
            return u, None, err
        t0 = time.time(); bad = 0
        with ThreadPoolExecutor(8) as ex, open(out, "w") as f:
            for u, res, err in ex.map(one, ids):
                if res is None: bad += 1; print("FAILED", u[-8:], err); continue
                lab, us = res
                usage["in"] += us.input_tokens; usage["out"] += us.output_tokens
                usage["cache_r"] += us.cache_read_input_tokens or 0; usage["cache_w"] += us.cache_creation_input_tokens or 0
                f.write(json.dumps({"StudyInstanceUID": u, **{KEYS[k]: v for k, v in lab.items()}}, ensure_ascii=False) + "\n")
        print(f"{model}: {n - bad}/{n} ok in {time.time() - t0:.0f}s | tokens {usage}")
    elif mode == "batch":
        model, out = sys.argv[2], sys.argv[3]
        R = reports(); c = client()
        b = c.messages.batches.create(requests=[Request(custom_id=u.replace(".", "_")[-64:], params=MessageCreateParamsNonStreaming(**params(model, R[u]))) for u in R.index])
        json.dump({"batch": b.id, "ids": {u.replace(".", "_")[-64:]: u for u in R.index}}, open(out, "w"))
        print("batch", b.id, b.processing_status, len(R), "requests")
    elif mode == "collect":
        meta, out = json.load(open(sys.argv[2])), sys.argv[3]
        c = client(); ok = bad = 0
        with open(out, "w") as f:
            for res in c.messages.batches.results(meta["batch"]):
                u = meta["ids"][res.custom_id]
                if res.result.type != "succeeded" or res.result.message.stop_reason != "end_turn": bad += 1; continue
                try: lab = parse(res.result.message)
                except Exception: bad += 1; continue
                f.write(json.dumps({"StudyInstanceUID": u, **{KEYS[k]: v for k, v in lab.items()}}, ensure_ascii=False) + "\n"); ok += 1
        print("collected", ok, "failed", bad)
