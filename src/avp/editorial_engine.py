"""Editorial engine v3: story selection, narrative design, bilingual writing and review."""
from __future__ import annotations
import json, os, re
from pathlib import Path
from . import brief as brief_mod
from . import factcheck
from .manifest import VideoProject
from .models import Script, Segment, dedupe_segments
from .log import get_logger
log=get_logger("avp.editorial")

DIRECTOR_SYSTEM="""You are the editorial director of a serious astronomy magazine. You do not write the script. Decide what story is worth telling. Prefer a strong tension, reversal, meaningful scale, real uncertainty, human/instrument history, mechanism, or consequence. Reject obvious trivia, generic facts, manufactured mystery, hype, and angles that cannot be told visually in a short video. Return STRICT JSON."""
DIRECTOR_USER="""TOPIC: {topic}\nFACT BASE:\n{facts}\nRECENT CHANNEL HISTORY:\n{history}\nGenerate exactly 8 genuinely different editorial angles. For each: angle, central_question, why, value_sources, strongest_fact, risk. Do not write script prose."""
SELECT_SYSTEM="""You are an independent senior editor at a serious astronomy magazine. Choose the strongest story angle from the candidates. Correctness alone is insufficient. Prefer a clear editorial idea with scientific substance, a reason to care, and a coherent 48-second story. Reject familiar trivia and manufactured mystery. Return STRICT JSON."""
SELECT_USER="""TOPIC: {topic}\nCANDIDATES:\n{candidates}\nReturn {{\"winner\":1,\"why_this_story\":\"...\",\"rejected\":[{{\"id\":2,\"reason\":\"...\"}}]}}"""
BRIEF_SYSTEM="""You are the editorial desk of a serious astronomy magazine. Turn the chosen angle into an editorial brief, not a script. Make explicit what the story is, what it is not, and what the audience should understand. Use only the supplied facts. Return STRICT JSON."""
BRIEF_USER="""TOPIC: {topic}\nCHOSEN ANGLE:\n{winner}\nEDITOR'S REASON:\n{why}\nFACT BASE:\n{facts}\nReturn {{\"editorial_angle\":\"...\",\"central_question\":\"...\",\"central_tension\":\"...\",\"audience_takeaway\":\"...\",\"key_facts\":[\"...\"],\"optional_facts\":[\"...\"],\"excluded_facts\":[\"...\"],\"misconception\":\"...\",\"quantitative_comparison\":\"...\",\"opening_type\":\"...\",\"closing_type\":\"...\",\"what_not_to_do\":[\"...\"]}}"""
NARRATIVE_SYSTEM="""You are a narrative designer for a serious astronomy magazine short. Using the brief, create THREE genuinely different narrative arcs for the SAME story. Each has 5-7 beats. A beat contains only the fact, visual idea and narrative role; do not write final prose. The arc must escalate and end by resolving or honestly opening the central question. Return STRICT JSON."""
NARRATIVE_USER="""TOPIC: {topic}\nBRIEF:\n{brief}\nFACT BASE:\n{facts}\nReturn {{\"arcs\":[{{\"id\":1,\"thesis\":\"...\",\"beats\":[{{\"beat\":1,\"fact\":\"...\",\"visual\":\"...\",\"role\":\"opening|build|turn|peak|close\"}}],\"why\":\"...\"}},{{\"id\":2,\"thesis\":\"...\",\"beats\":[...],\"why\":\"...\"}},{{\"id\":3,\"thesis\":\"...\",\"beats\":[...],\"why\":\"...\"}}]}}"""
ARC_SELECT_SYSTEM="""You are a senior narrative editor. Choose the strongest arc for a short astronomy-magazine video. Judge story coherence, escalation, distinct beats, scientific integrity, visual feasibility and ending. Return STRICT JSON."""
ARC_SELECT_USER="""TOPIC: {topic}\nBRIEF:{brief}\nARCS:{arcs}\nReturn {{\"winner\":1,\"why\":\"...\"}}"""
WRITE_SYSTEM="""You are an astronomy-magazine writer. Write finished short-form prose from the supplied editorial brief and chosen arc. This is not school science, generic YouTube copy or ad copy. Assume basic astronomy literacy. Priorities: precision, specificity, information density, depth, natural spoken rhythm. Every sentence must inform, contextualise or advance the story. Use concrete nouns and meaningful numbers. Never invent a mechanism, number, date, status, superlative or attribution. Never repeat a fact. Avoid AI filler such as amazing, incredible, fascinating, mysterious, journey, unlock, delve, breathtaking, mind-blowing. No generic rhetorical hooks. No generic CTA. Keep the exact beat count. Return STRICT JSON."""
WRITE_USER="""LANGUAGE: {language}\nTOPIC: {topic}\nBRIEF:\n{brief}\nCHOSEN ARC:\n{arc}\nFACT BASE:\n{facts}\nReturn {{\"title\":\"...\",\"segments\":[{{\"narration\":\"...\",\"visual\":\"...\",\"keywords\":[\"...\"]}}],\"bridge_kind\":\"shoot|principle|none\",\"cta_bridge\":\"...\"}}\nWrite independently in this language from the same facts and beats; do not translate another language."""
REVIEW_SYSTEM="""You are a ruthless independent editor at a serious astronomy magazine. Diagnose the script; do not rewrite it. A script may be factually correct and still be editorially weak. Compare it with the brief and benchmark. Rate each dimension weak/solid/strong/exceptional: idea, specificity, density, originality, narration, language. Rate ai_smell none/mild/strong/grave. List every sentence that prevents a stronger result, with a precise reason. Flag fact-risk, repetition, generic promotion and misleading visuals. Return STRICT JSON."""
REVIEW_USER="""TOPIC:{topic}\nBRIEF:{brief}\nBENCHMARK:{benchmark}\nSCRIPT:{script}\nReturn {{\"dimensions\":{{\"idea\":\"...\",\"specificity\":\"...\",\"density\":\"...\",\"originality\":\"...\",\"narration\":\"...\",\"language\":\"...\",\"ai_smell\":\"...\"}},\"weak_sentences\":[{{\"segment\":1,\"quote\":\"...\",\"reason\":\"...\"}}],\"fact_risks\":[\"...\"],\"decision\":\"publish|rewrite|reject_story\",\"summary\":\"...\"}}"""
REVISE_SYSTEM=WRITE_SYSTEM+"\nYou are revising after an editorial diagnosis. Fix the diagnosed weaknesses without flattening the voice or changing the chosen story."

def _json(text):
    text=(text or "").strip(); text=re.sub(r"^```(?:json)?\s*|\s*```$","",text,flags=re.I|re.S).strip()
    try:return json.loads(text)
    except json.JSONDecodeError:
        m=re.search(r"\{.*\}",text,re.S)
        if not m: raise
        return json.loads(m.group(0))

def _api(cfg, editor=False):
    if editor:
        key=os.getenv("AVP_EDITOR_API_KEY","").strip() or factcheck._api_key(cfg)
        url=os.getenv("AVP_EDITOR_URL","https://api.deepseek.com/chat/completions").strip()
        model=os.getenv("AVP_EDITOR_MODEL","").strip() or str(getattr(cfg.script,"factcheck_model","deepseek-chat"))
    else:
        key=factcheck._api_key(cfg); url="https://api.deepseek.com/chat/completions"; model=str(getattr(cfg.script,"brief_model","") or getattr(cfg.script,"factcheck_model","deepseek-chat"))
    return key,url,model

def _call(cfg,system,user,editor=False,temp=.2,max_tokens=3000):
    key,url,model=_api(cfg,editor)
    if not key: raise RuntimeError("No editorial API key: set DEEPSEEK_API_KEY or AVP_EDITOR_API_KEY")
    r=__import__('requests').post(url,headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},json={"model":model,"messages":[{"role":"system","content":system},{"role":"user","content":user}],"temperature":temp,"response_format":{"type":"json_object"},"max_tokens":max_tokens},timeout=(10,180))
    if r.status_code>=400: raise RuntimeError(f"editorial model HTTP {r.status_code}: {(r.text or '')[:300]}")
    return _json(r.json()["choices"][0]["message"]["content"])

def _history(cfg):
    root=Path(cfg.paths.projects_dir).expanduser(); rows=[]
    for p in sorted(root.glob("*/manifest.json"))[-60:]:
        try:
            d=json.loads(p.read_text()); rows.append(f"- {d.get('topic','')} | {d.get('title','')}")
        except Exception: pass
    return "\n".join(rows) or "(none)"

def _facts(topic,cfg,project):
    text=brief_mod.build(topic,cfg,out_dir=project.root)
    if not text: raise RuntimeError("No audited fact sheet available; refusing to invent facts")
    raw={}
    try: raw=json.loads((project.root/"brief.json").read_text())
    except Exception: pass
    return text,raw

def _benchmark():
    p=Path("editorial/benchmark.json")
    if not p.exists(): return "No benchmark seeded yet. Judge against the editorial standard in the prompt."
    try:return json.dumps(json.loads(p.read_text()),ensure_ascii=False,indent=2)
    except Exception:return "Benchmark unavailable."

def _script(data,topic,target):
    segs=[]
    for i,row in enumerate(data.get("segments") or [],1):
        if isinstance(row,dict) and str(row.get("narration","")).strip(): segs.append(Segment(i,str(row["narration"]).strip(),str(row.get("visual","")).strip(),[str(k).strip() for k in (row.get("keywords") or []) if str(k).strip()]))
    segs=dedupe_segments(segs)
    if not 5<=len(segs)<=7: raise RuntimeError(f"editorial script must contain 5-7 content beats, got {len(segs)}")
    return Script(title=str(data.get("title") or topic).strip(),segments=segs,target_seconds=target,topic=topic,cta_bridge=str(data.get("cta_bridge") or "").strip(),bridge_kind=str(data.get("bridge_kind") or "none").strip().lower())

def _publishable(r):
    d=r.get("dimensions") or {}
    return str(r.get("decision","")).lower()=="publish" and all(str(d.get(k,"weak")).lower() in {"solid","strong","exceptional"} for k in ("idea","specificity","density","narration","language")) and str(d.get("originality","weak")).lower() in {"strong","exceptional"} and str(d.get("ai_smell","grave")).lower() in {"none","mild"}

def _story_rejected(r):
    d=r.get("dimensions") or {}
    return str(r.get("decision","")).lower()=="reject_story" or str(d.get("idea","")).lower()=="weak" or str(d.get("originality","")).lower()=="weak"

def _pass(cfg,topic,project,facts,history,attempt):
    director=_call(cfg,DIRECTOR_SYSTEM,DIRECTOR_USER.format(topic=topic,facts=facts,history=history),False,.55,3200)
    angles=director.get("angles") or director.get("candidates") or []
    if len(angles)<3: raise RuntimeError("director returned fewer than 3 angles")
    selection=_call(cfg,SELECT_SYSTEM,SELECT_USER.format(topic=topic,candidates=json.dumps(angles,ensure_ascii=False,indent=2)),True,.1,1800)
    try:i=int(selection.get("winner",1))-1
    except: i=0
    i=max(0,min(i,len(angles)-1)); winner=angles[i]
    brief=_call(cfg,BRIEF_SYSTEM,BRIEF_USER.format(topic=topic,winner=json.dumps(winner,ensure_ascii=False,indent=2),why=selection.get("why_this_story",""),facts=facts),False,.1,2600)
    (project.root/"editorial_director.json").write_text(json.dumps({"attempt":attempt,"angles":angles,"selection":selection},indent=2,ensure_ascii=False))
    (project.root/"editorial_brief.json").write_text(json.dumps(brief,indent=2,ensure_ascii=False))
    arcs_data=_call(cfg,NARRATIVE_SYSTEM,NARRATIVE_USER.format(topic=topic,brief=json.dumps(brief,ensure_ascii=False,indent=2),facts=facts),False,.5,3500)
    arcs=arcs_data.get("arcs") or []
    if len(arcs)<3: raise RuntimeError("narrative designer returned fewer than 3 arcs")
    arc_pick=_call(cfg,ARC_SELECT_SYSTEM,ARC_SELECT_USER.format(topic=topic,brief=json.dumps(brief,ensure_ascii=False,indent=2),arcs=json.dumps(arcs,ensure_ascii=False,indent=2)),True,.1,900)
    try:ai=int(arc_pick.get("winner",1))-1
    except: ai=0
    ai=max(0,min(ai,len(arcs)-1)); arc=arcs[ai]
    (project.root/"narrative_arcs.json").write_text(json.dumps({"arcs":arcs,"selection":arc_pick},indent=2,ensure_ascii=False))
    target=int(cfg.script.target_seconds)
    en=_script(_call(cfg,WRITE_SYSTEM,WRITE_USER.format(language="English",topic=topic,brief=json.dumps(brief,ensure_ascii=False,indent=2),arc=json.dumps(arc,ensure_ascii=False,indent=2),facts=facts),False,.55,3000),topic,target)
    it=_script(_call(cfg,WRITE_SYSTEM,WRITE_USER.format(language="Italian",topic=topic,brief=json.dumps(brief,ensure_ascii=False,indent=2),arc=json.dumps(arc,ensure_ascii=False,indent=2),facts=facts),False,.55,3000),topic,target)
    if len(en.segments)!=len(it.segments): raise RuntimeError("EN/IT writers returned different beat counts")
    for a,b in zip(en.segments,it.segments): b.visual=a.visual; b.keywords=list(a.keywords)
    en.cta_bridge=str((en.cta_bridge or "")).strip(); en.bridge_kind=en.bridge_kind or "none"
    it.cta_bridge=en.cta_bridge; it.bridge_kind=en.bridge_kind
    benchmark=_benchmark()
    enrev=_call(cfg,REVIEW_SYSTEM,REVIEW_USER.format(topic=topic,brief=json.dumps(brief,ensure_ascii=False,indent=2),benchmark=benchmark,script=json.dumps(en.to_dict(),ensure_ascii=False,indent=2)),True,.05,2800)
    itrev=_call(cfg,REVIEW_SYSTEM,REVIEW_USER.format(topic=topic,brief=json.dumps(brief,ensure_ascii=False,indent=2),benchmark=benchmark,script=json.dumps(it.to_dict(),ensure_ascii=False,indent=2)),True,.05,2800)
    (project.root/"editorial_review_en_v1.json").write_text(json.dumps(enrev,indent=2,ensure_ascii=False)); (project.root/"editorial_review_it_v1.json").write_text(json.dumps(itrev,indent=2,ensure_ascii=False))
    for s,r,lang in ((en,enrev,"English"),(it,itrev,"Italian")):
        if _publishable(r): continue
        if _story_rejected(r): raise RuntimeError("STORY_REJECTED")
        revised=_call(cfg,REVISE_SYSTEM,WRITE_USER.format(language=lang,topic=topic,brief=json.dumps(brief,ensure_ascii=False,indent=2),arc=json.dumps(arc,ensure_ascii=False,indent=2),facts=facts)+"\nEDITOR DIAGNOSIS:\n"+json.dumps(r,ensure_ascii=False,indent=2),False,.35,3000)
        ns=_script(revised,topic,target)
        if len(ns.segments)!=len(s.segments): raise RuntimeError(f"{lang} rewrite changed beat count")
        s.title=ns.title; s.segments=ns.segments; s.cta_bridge=ns.cta_bridge; s.bridge_kind=ns.bridge_kind
        rr=_call(cfg,REVIEW_SYSTEM,REVIEW_USER.format(topic=topic,brief=json.dumps(brief,ensure_ascii=False,indent=2),benchmark=benchmark,script=json.dumps(s.to_dict(),ensure_ascii=False,indent=2)),True,.05,2800)
        suffix="en" if lang=="English" else "it"; (project.root/f"editorial_review_{suffix}_v2.json").write_text(json.dumps(rr,indent=2,ensure_ascii=False))
        if not _publishable(rr): raise RuntimeError(f"{lang} script below editorial standard after rewrite")
    # Finalise both languages on the same beat structure. The English Script carries the Italian subtitle text
    # because the existing captions/voice stages consume script.json. No translation pass is allowed here.
    for a,b in zip(en.segments,it.segments):
        a.italian=b.narration
        b.visual=a.visual; b.keywords=list(a.keywords)
    if cfg.funnel.enabled:
        bridge=en.cta_bridge.strip(); kind=en.bridge_kind
        if kind=="none": cta=bridge if str(getattr(cfg.funnel,"bridge_policy","always")).lower()=="honest" and bridge else cfg.funnel.cta_line.format(app=cfg.funnel.app_name)
        elif bridge: cta=f"{bridge} Get {cfg.funnel.app_name} — link in bio."
        else: cta=cfg.funnel.cta_line.format(app=cfg.funnel.app_name)
        en.segments.append(Segment(len(en.segments)+1,cta,"App endcard",[],kind="cta"))
        it.segments.append(Segment(len(it.segments)+1,cta,"App endcard",[],kind="cta"))
    return en,it,brief,{"director":angles,"selection":selection,"arc_selection":arc_pick,"en_review":enrev,"it_review":itrev}

def stage_script(project: VideoProject,cfg,topic: str|None)->Script:
    topic=(topic or str(project.manifest.data.get("topic") or "")).strip()
    if not topic: raise ValueError("No topic given and no existing one. Pass --topic.")
    facts,_=_facts(topic,cfg,project); history=_history(cfg)
    last=None
    for attempt in (1,2):
        try:
            en,it,brief,evidence=_pass(cfg,topic,project,facts,history,attempt)
            project.script_json.write_text(json.dumps(en.to_dict(),indent=2,ensure_ascii=False))
            (project.root/"italian_script.json").write_text(json.dumps(it.to_dict(),indent=2,ensure_ascii=False))
            from . import stages
            stages.emit_script_md(en,project.script_md)
            project.manifest.data["topic"]=topic; project.manifest.data["title"]=en.title
            project.manifest.data.setdefault("editorial",{}).update({"version":3,"autonomous":True,"editor_model":_api(cfg,True)[2],"fact_base":"audited brief","benchmark":str(Path("editorial/benchmark.json"))})
            project.manifest.mark("script","done",title=en.title,segments=len(en.segments)); project.manifest.save()
            log.info("Editorial script approved: %r",en.title)
            return en
        except RuntimeError as e:
            last=e
            if str(e)=="STORY_REJECTED" and attempt==1:
                log.warning("Story rejected; selecting a new editorial angle")
                continue
            raise
    raise last or RuntimeError("editorial generation failed")
