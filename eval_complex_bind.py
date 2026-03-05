"""Unified Evaluation: LLM Parser + Adaptive Merging + Orthogonal Disentanglement
Evaluates all prompts from complexbind dataset."""

# Configuration options
USLLMPRSR = False  #False to use SpaCy parser instead
USADPTVMERGN = True  # False to diable adaptive token merging
USRTHGNLDISNTNGLMNT = False  # STrue to enable orthogonal disentanglement

import sys
import os
import gc
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

bsd = Path(__file__).parent.absolute()

if str(bsd) not in sys.path:
    sys.path.insert(0, str(bsd))

from parsers import LLMParser, SpaCyParser
from pipe_tome import tomePipeline
from utils import ptp_utils


# complex_bind directory resolution
CMPLXBNDDR = bsd / "complex_bind"
if not CMPLXBNDDR.exists():
    raise FileNotFoundError(f"Could not find complex_bind directory at {CMPLXBNDDR}")


# Prompt loading
def loadprompts(flpth, lmt=None):
    flpth = Path(flpth)

    if not flpth.exists():
        return []

    prmts: List[str] = []

    with open(flpth, "r", encoding="utf-8") as f:
        for ln in f:
            clnd = ln.strip()
            if not clnd:
                continue

            prmts.append(clnd)

            if lmt is not None and len(prmts) >= lmt:
                break

    return prmts


# Prompt merging
def extractmergedprompt(prmpt: str) -> str:
    wrds = prmpt.split()
    bjcts: List[str] = []

    for i, w in enumerate(wrds):
        if w in ("a", "an") and i + 1 < len(wrds):
            lkhdnd = min(i + 5, len(wrds))
            j = lkhdnd

            for k in range(i + 1, lkhdnd):
                if wrds[k] in ("and", "a", "an", ","):
                    j = k
                    break

            if j > i + 1:
                bjcts.append(wrds[j - 1])

    if len(bjcts) >= 2:
        return f"a {bjcts[0]} and a {bjcts[1]}"

    return prmpt



def filtertextllm(tknndcs, prmptnchr):
    fltrd = [(idx, p) for idx, p in zip(tknndcs, prmptnchr) if len(idx[0]) > 0 or len(idx[1]) > 0]
    return [f[0] for f in fltrd], [f[1] for f in fltrd]


def generatevqaquestions(prmpt, ctgry):
    wrds, qstns = prmpt.lower().split(), []
    skp = ['and', 'a', 'an', 'the']
    
    if ctgry == "Color":
        # common colors we care about
        clrs = ['white', 'black', 'red', 'blue', 'green', 'yellow', 'brown', 'gold', 'silver', 'pink', 'purple', 'orange', 'gray', 'grey']
        for i, w in enumerate(wrds):
            if w in clrs:
                # try to find the object being described
                bj = None
                for j in range(i+1, min(i+4, len(wrds))):
                    if wrds[j] not in skp and len(wrds[j]) > 2:
                        bj = wrds[j]
                        break
                
                if bj:
                    qstns.append({"question": f"What color is the {bj}?", "expected": [w]})
                    qstns.append({"question": f"Is the {bj} {w}?", "expected": ["yes", w]})
                    break
                    
    elif ctgry == "Texture":
        # material types
        txtrs = {'wooden': ['wood', 'wooden'], 'glass': ['glass'], 'metallic': ['metal', 'metallic'],
                   'fluffy': ['fluffy', 'soft', 'fur'], 'leather': ['leather'], 'plastic': ['plastic'],
                   'rubber': ['rubber'], 'fabric': ['fabric', 'cloth']}
        
        for i, w in enumerate(wrds):
            if w in txtrs:
                bj = None
                for j in range(i+1, min(i+4, len(wrds))):
                    if wrds[j] not in skp and len(wrds[j]) > 2:
                        bj = wrds[j]
                        break
                        
                if bj:
                    # hacky way to clean up adjective forms
                    clean_w = w.replace('ic', '').replace('y', '')
                    qstns.append({"question": f"Is the {bj} made of {clean_w}?", "expected": ["yes"] + txtrs[w]})
                    qstns.append({"question": f"What material is the {bj}?", "expected": txtrs[w]})
                    break
                    
    elif ctgry == "Shape":
        shps = {'round': ['round', 'circular', 'circle'], 'square': ['square', 'rectangular'],
                 'triangular': ['triangular', 'triangle'], 'spherical': ['spherical', 'round', 'ball'],
                 'cylindrical': ['cylindrical', 'cylinder'], 'rectangular': ['rectangular', 'rectangle', 'square'],
                 'oval': ['oval', 'elliptical', 'round'], 'cubic': ['cubic', 'cube', 'square'],
                 'conical': ['conical', 'cone'], 'pentagonal': ['pentagonal', 'pentagon'],
                 'pyramidal': ['pyramidal', 'pyramid', 'triangular'], 'oblong': ['oblong', 'elongated', 'long'],
                 'teardrop': ['teardrop', 'drop'], 'diamond': ['diamond'], 'crescent': ['crescent', 'curved']}
        
        for i, w in enumerate(wrds):
            if w in shps:
                bj = None
                for j in range(i+1, min(i+4, len(wrds))):
                    if wrds[j] not in skp and len(wrds[j]) > 2:
                        bj = wrds[j]
                        break
                if bj:
                    qstns.append({"question": f"Is the {bj} {w}?", "expected": ["yes"] + shps[w]})
                    qstns.append({"question": f"What shape is the {bj}?", "expected": shps[w]})
                    break
                    
    elif ctgry == "Complex":
        # words to ignore when looking for objects
        skpwrds = ['a', 'an', 'the', 'and', 'with', 'that', 'is', 'are', 'was', 'were', 'has', 'have', 'had',
                     'chasing', 'saw', 'held', 'served', 'looked', 'without', 'except', 'behind', 'next', 'on', 'in', 'at']
        
        # extract potential objects (just grab first few substantive words)
        bjcts = []
        for w in wrds:
            if w not in skpwrds and len(w) > 2:
                bjcts.append(w)
            if len(bjcts) >= 3:
                break
        
        for bj in bjcts:
            qstns.append({"question": f"Is there a {bj} in the image?", "expected": ["yes", bj]})
            qstns.append({"question": "What is in the image?", "expected": [bj] + bjcts[:2]})
            
    return qstns

def evaluateimage(mg, qstns, processor, mdl, dvc):
    if not qstns:
        return 0.0, []
    crrct, dtls = 0, []
    for q in qstns:
        npts = processor(mg, q["question"], return_tensors="pt").to(dvc)
        with torch.no_grad():
            nswr = processor.decode(mdl.generate(**npts, max_length=20)[0], skip_special_tokens=True).lower()
        scrct = any(exp in nswr for exp in q["expected"])
        if scrct:
            crrct += 1
        dtls.append({"question": q["question"], "answer": nswr, "expected": q["expected"], "correct": scrct})
    return crrct / len(qstns), dtls


def loadsdxlmodel(dvc):
    mdl = tomePipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(dvc)
    mdl.unet.requires_grad_(False)
    mdl.vae.requires_grad_(False)
    return mdl


def main(usllmprsr=True):
    """
    Main evaluation function.
    
    Args:
        use_llm_parser: If True, use LLM parser (Qwen), if False, use SpaCy parser
    """
    dvc = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dvc}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    
    prsrn = "llm" if USLLMPRSR else "spacy"
    tptbs = bsd / f"complex_bind_eval_results_{prsrn}"
    (tptbs / "images").mkdir(parents=True, exist_ok=True)
    
    prmptfls = {
        "Color": CMPLXBNDDR / "color_prompts.txt",
        "Shape": CMPLXBNDDR / "shape_prompts.txt",
        "Texture": CMPLXBNDDR / "texture_prompts.txt",
        "AdaptiveMerge": CMPLXBNDDR / "adaptive_merge_prompts.txt",
        "Complex": CMPLXBNDDR / "llm_parser_prompts.txt",
    }
    
    llprmpts = [(ctgry, loadprompts(f)) for ctgry, f in prmptfls.items()]
    rslt = {ctgry: {"scores": [], "prompts": [], "details": []} for ctgry in prmptfls.keys()}
    ttlprmpts = sum(len(p) for _, p in llprmpts)
    print(f"\nLoaded {ttlprmpts} prompts from complex_bind")
    
    # Phase 1: Parse all prompts
    print("\n" + "="*70)
    if USLLMPRSR:
        print("PHASE 1: Parsing prompts with LLM")
        print("="*70)
        llmprsr = LLMParser(modelpath="stabilityai/stable-diffusion-xl-base-1.0", llmname="Qwen/Qwen2.5-14B-Instruct")
        llmprsr.loadllm()
        
        prsddt = []
        ttlprsd = 0
        for ctgry, prmpts in llprmpts:
            for idx, prmpt in enumerate(prmpts):
                ttlprsd += 1
                try:
                    smntcgrps = llmprsr.llmsemanticparsing(prmpt)
                    tknndcs = llmprsr.alignsemanticgrps(smntcgrps, prmpt)
                    prmptnchr = [g["phrase"] for g in smntcgrps["groups"]]
                    tknndcs, prmptnchr = filtertextllm(tknndcs, prmptnchr)
                    prsddt.append((ctgry, idx, prmpt, extractmergedprompt(prmpt), tknndcs, prmptnchr))
                except Exception as e:
                    print(f"  Error parsing: {e}")
                    prsddt.append((ctgry, idx, prmpt, extractmergedprompt(prmpt), [], []))
                if ttlprsd % 50 == 0:
                    print(f"  Parsed {ttlprsd}/{ttlprmpts}")
        
        llmprsr.unloadllm()
        if dvc.startswith("cuda"):
            torch.cuda.empty_cache()
            gc.collect()
        print("LLM unloaded")
    else:
        print("PHASE 1: Parsing prompts with SpaCy")
        print("="*70)
        spcyprsr = SpaCyParser(model_path="stabilityai/stable-diffusion-xl-base-1.0")
        
        prsddt = []
        ttlprsd = 0
        for ctgry, prmpts in llprmpts:
            for idx, prmpt in enumerate(prmpts):
                ttlprsd += 1
                try:
                    tknndcs, prmptnchr = spcyprsr.parse(prmpt)
                    prsddt.append((ctgry, idx, prmpt, extractmergedprompt(prmpt), tknndcs, prmptnchr))
                except Exception as e:
                    print(f"  Error parsing: {e}")
                    prsddt.append((ctgry, idx, prmpt, extractmergedprompt(prmpt), [], []))
                if ttlprsd % 50 == 0:
                    print(f"  Parsed {ttlprsd}/{ttlprmpts}")
    
    # Phase 2: Generate images
    print("\n" + "="*70)
    print("PHASE 2: Generating images")
    print("="*70)
    mdl = loadsdxlmodel(dvc)
    blpprcssr = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    blpmdl = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(dvc)
    blpmdl.eval()
    
    cnfg = {
        "n_inference_steps": 50, "guidance_scale": 7.5, "attention_res": 32,
        "thresholds": {i: 26 - i*0.5 if i < 5 else 21 for i in range(10)},
        "tome_control_steps": [5, 5], "token_refinement_steps": 3, "attention_refinement_steps": [4, 4],
        "eot_replace_step": 0, "use_pose_loss": False, "scale_factor": 3, "scale_range": (1.0, 0.0)
    }
    
    ctgrycnts = {ctgry: sum(1 for p in prsddt if p[0] == ctgry) for ctgry in prmptfls.keys()}
    ctgryprcssd = {ctgry: 0 for ctgry in prmptfls.keys()}
    bssd = 42
    
    strttm = datetime.now()
    for ttlprcssd, (ctgry, idx, prmpt, prmptmrgd, tknndcs, prmptnchr) in enumerate(prsddt, 1):
        ctgryprcssd[ctgry] += 1
        print(f"\n[{ctgryprcssd[ctgry]}/{ctgrycnts[ctgry]}] ({ttlprcssd}/{ttlprmpts}) {prmpt[:60]}...")
        
        vqct = "Texture" if ctgry == "AdaptiveMerge" else ctgry
        qstns = generatevqaquestions(prmpt, vqct)
        if ctgry == "AdaptiveMerge" and not qstns:
            qstns = generatevqaquestions(prmpt, "Color") or generatevqaquestions(prmpt, "Complex")
        
        try:
            rnstndrdsd = len(tknndcs) == 0
            g = torch.Generator(dvc).manual_seed(bssd + ttlprcssd)
            cntrllr = ptp_utils.AttentionStore()
            ptp_utils.register_attention_control(mdl, cntrllr)
            
            tpts = mdl(prompt=prmpt, guidance_scale=cnfg["guidance_scale"], generator=g,
                          num_inference_steps=cnfg["n_inference_steps"], attention_store=cntrllr,
                          indices_to_alter=tknndcs, prompt_anchor=prmptnchr,
                          attention_res=cnfg["attention_res"], run_standard_sd=rnstndrdsd,
                          thresholds=cnfg["thresholds"], scale_factor=cnfg["scale_factor"],
                          scale_range=cnfg["scale_range"], prompt3=prmptmrgd,
                          prompt_length=len(prmpt.split()), token_refinement_steps=cnfg["token_refinement_steps"],
                          attention_refinement_steps=cnfg["attention_refinement_steps"],
                          tome_control_steps=cnfg["tome_control_steps"], eot_replace_step=cnfg["eot_replace_step"],
                          use_pose_loss=cnfg["use_pose_loss"], use_adaptive_merging=USADPTVMERGN,
                          use_orthogonal_disentanglement=USRTHGNLDISNTNGLMNT,
                          negative_prompt="low res, ugly, blurry, hybrid, mutant, merged bodies")
            
            mg = tpts.images[0]
            mg.save(tptbs / "images" / f"{ctgry}_{idx}_{prmpt.replace(' ', '_')[:40]}.png")
            
            if qstns:
                scr, dtls = evaluateimage(mg, qstns, blpprcssr, blpmdl, dvc)
                rslt[ctgry]["scores"].append(scr)
                rslt[ctgry]["prompts"].append(prmpt)
                rslt[ctgry]["details"].append({"prompt": prmpt, "score": scr, "questions": dtls})
                print(f"  Score: {scr:.2f}")
            else:
                rslt[ctgry]["scores"].append(None)
                rslt[ctgry]["prompts"].append(prmpt)
        except Exception as e:
            print(f"  ERROR: {e}")
            rslt[ctgry]["scores"].append(None)
            rslt[ctgry]["prompts"].append(prmpt)
            rslt[ctgry]["details"].append({"prompt": prmpt, "error": str(e)})
        
        if ttlprcssd % 20 == 0:
            saveresults(rslt, tptbs, False)
    
    saveresults(rslt, tptbs, True)
    printsummary(rslt, strttm, datetime.now(), tptbs, list(prmptfls.keys()))


def saveresults(rslt, tptdr, sfnl):
    sffx = "_final" if sfnl else "_checkpoint"
    with open(tptdr / f"results{sffx}.json", "w") as f:
        json.dump(rslt, f, indent=2)
    
    prsrn = "Qwen 2.5-14B-Instruct" if USLLMPRSR else "SpaCy"
    mrgnstts = "Adaptive" if USADPTVMERGN else "None"
    dsntnglmntstts = "Orthogonal" if USRTHGNLDISNTNGLMNT else "None"
    with open(tptdr / f"summary{sffx}.txt", "w") as f:
        f.write(f"{prsrn} Parser Results\n" + "="*70 + "\n")
        f.write(f"Parser: {prsrn} | Merging: {mrgnstts} | Disentanglement: {dsntnglmntstts}\n")
        f.write("Source: complex_bind directory\n\n")
        llscrs = []
        for ctgry in rslt.keys():
            scrs = [s for s in rslt[ctgry]["scores"] if s is not None]
            mean = np.mean(scrs) if scrs else 0.0
            llscrs.extend(scrs)
            f.write(f"{ctgry}: {mean:.4f} (n={len(scrs)})\n")
        f.write(f"\nOverall: {np.mean(llscrs):.4f} (n={len(llscrs)})\n")


def printsummary(rslt, strttm, ndtm, tptbs, ctgrs):
    prsrn = "Qwen 2.5-14B-Instruct" if USLLMPRSR else "SpaCy"
    mrgnstts = "Adaptive" if USADPTVMERGN else "None"
    dsntnglmntstts = "Orthognal" if USRTHGNLDISNTNGLMNT else "None"
    print("\n" + "="*70)
    print(f"RESULTSSUMMARY - {prsrn} Parser")
    print(f"Parser: {prsrn} | Merging: {mrgnstts} | Disentanglement: {dsntnglmntstts}")
    print("="*70)
    print(f"Started: {strttm.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {ndtm.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {ndtm - strttm}\n")
    print(f"{'Category':<15} | {'Mean Score':<12} | {'N':<6}")
    print("-" * 38)
    llscrs = []
    for ctgry in ctgrs:
        scrs = [s for s in rslt[ctgry]["scores"] if s is not None]
        mean = np.mean(scrs) if scrs else 0.0
        llscrs.extend(scrs)
        print(f"{ctgry:<15} | {mean:.4f}       | {len(scrs)}")
    print("-" * 38)
    print(f"{'OVERALL':<15} | {np.mean(llscrs):.4f}       | {len(llscrs)}")
    print("="*70)
    print(f"\nResults: {tptbs.absolute()}")


if __name__ == "__main__":
    main()
