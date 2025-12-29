"""
Automates the annotation of veterinary thoracic radiography reports using a local LLM with few-shot prompting.
Processes report texts to compute term frequencies and optionally generate a word cloud.
Classifies each report into Pleura, Intrapulmonar, Ambos, or Ausência.
Exports a one-hot CSV linking report texts to their corresponding PDFs.
Designed to accelerate dataset labeling for downstream anomaly detection experiments.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.utils import logging as hf_logging
from typing import List, Iterable, Tuple
from dataclasses import dataclass
from wordcloud import WordCloud
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import unicodedata
import torch
import csv
import os
import re

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
hf_logging.set_verbosity_error()

TEXT_DIR = Path("data") / "text"
PDF_DIR = Path("data")
PATTERN = re.compile(r"^radiografia[\s_-]*torax")

OUT_DIR = Path("output")
FREQ_CSV = OUT_DIR / "frequencias.csv"
WORDCLOUD_PNG = OUT_DIR / "wordcloud.png"
CSV_CLASS = OUT_DIR / "classificacao_opacidades.csv"
CSV_CATS = ["pleura", "intrapulmonar", "ambos", "ausencia"]

MODEL_ROOT = Path("/scratch/victoria.estanislau/.cache/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B")
MAX_TOKENS_IN = 2048
MAX_NEW_TOKENS = 6

_re_token = re.compile(r"\b[\w]+\b", re.UNICODE)

STOP_PHRASES = {"raio x", "belo horizonte"}
STOP_TERMS_SINGLE = {
    "relatório","paciente","número","proprietário","sexo","peso","espécie","CRMV","MG",
    "Suspeita","Clínica","raça","veterinário","responsável","animal","comentários","pelagem",
    "tutor","(a)","femea","macho","nascimento","tel","celular","radiográfico","região",
    "interesse","achados","radriografia","Bruno","Ferrante","Hospital","UFMG","Pampulha",
    "Carlos","Luz","CEP","impressão","diagnóstica","diagnóstico","fone","incidêcias",
    "observações", "não", "com", "nao", "exame", "exames", "torax", "imagem", "radiograficos",
    "presenca", "laboratoriais", "clinicos", "outros", "para", "melhor", "capaz", "possui", "informações",
    "devido", "dados", "data", "pelo", "conjunto", "interpretar", "silhueta", "medico", "devem", "deve"
}

FEW_SHOT_EXAMPLES = {
    "Ambos": """
Achados Radiográficos:

Campos pulmonares apresentando opacificação intersticioalveolar multifocal. Apenas em projeção laterolateral direita, não visível em projeção ortogonal, visibiliza-se uma área nodular, de margens parcialmente definidas e de densidade de tecidos moles, medindo cerca de 0,86 cm de diâmetro, no primeiro espaço intercostal. Visibiliza-se fissura interlobar entre lobo cranial e médio direito em projeção ventrodorsal. Silhueta cardíaca: Formato usual, com contornos e dimensões preservados (VHS: 10,5 v). Traqueia: Trajeto e lúmen preservados, sem sinais de estreitamento ou opacificação. Esôfago: Não visibilizado, devido à ausência de conteúdo. Mediastino: Preservado. Diafragma, arcos costais e esterno: Mineralização das junções costocondrais e proliferações ósseas em margens dorsais das esternébras V e VI.

Observação:

Silhueta hepática ultrapassando os limites do gradil costal (hepatomegalia). Sugere-se ultrassonografia abdominal caso o médico veterinário responsável julgue necessário.

Impressão Diagnóstica:

Os achados radiográficos podem estar relacionados a pneumonia, não sendo possível descartar processo neoplásico associado, e efusão pleural.

Comentários:

Nódulos pulmonares menores que 5 mm podem não ser visibilizados no exame radiográfico. O colapso de traqueia é um processo dinâmico. A ausência de estreitamento do lúmen no momento do exame radiográfico não exclui a sua presença. O exame radiográfico não possui valor diagnóstico absoluto. As informações fornecidas devem ser confrontadas com dados clínicos, laboratoriais e com outros exames de imagem anteriores e/ou subsequentes. Somente o Médico Veterinário responsável pelo paciente é capaz de interpretar o conjunto de todas as informações.
""",
    "Intrapulmonar": """
Achados Radiográficos:

Campos pulmonares: Visibiliza-se padrão estruturado nodular difusamente distribuído em todos os lobos pulmonares, a maior nodulação medindo aproximadamente 3,63cmx3,75 cm.

Silhueta cardíaca: Formato usual, com contornos e dimensões preservados. Parcialmente obliterada em projeção laterolateral, devido à sobreposição de estruturas nodulares presentes no parênquima pulmonar.

Traqueia: Trajeto e lúmen preservados.

Esôfago: Não visibilizado, devido à ausência de conteúdo.

Mediastino: Preservado.

Diafragma, arcos costais e esterno: Mineralização de cartilagens costocondrais.

Obs.:

Presença de neoformações ósseas em cabeças umerais e em aspecto caudal das superfícies articulares escápuloumerais (entesófitos).

Impressão Diagnóstica:

Os achados radiográficos são compatíveis com metástase pulmonar.

O exame radiográfico não possui valor diagnóstico absoluto. As informações fornecidas devem ser confrontadas com dados clínicos, laboratoriais e com outros exames de imagem anteriores e/ou subsequentes. Somente o Médico Veterinário responsável pelo paciente é capaz de interpretar o conjunto de todas as informações.
""",
    "Pleura": """
Achados Radiográficos:

Nota-se aumento de opacificação de tecidos moles generalizada em tórax.

Campos pulmonares: Deslocados dorsalmente. Características radiográficas preservadas em porções passíveis de avaliação.

Silhueta cardíaca: Não caracterizada (sinal de silhueta).

Traqueia: Trajeto e lúmen preservados, sem sinal de estreitamento/opacificação. Mineralização de anéis cartilaginosos.

Mediastino: Preservado.

Diafragma: Não caracterizado (podendo estar obliterado pela presença de líquido no espaço pleural).

Arcos costais e esterno: Mineralização de cartilagens costocondrais.

Obs.:

Presença de acentuadas neoformações ósseas em bordas ventrais de placas terminais de vértebras T5-T6 e segmento vertebral de T11-L3 (espondilose).

Impressão Diagnóstica:

Os achados radiográficos podem estar relacionados com efusão pleural.

Comentários:

Nódulos pulmonares menores que 5 mm podem não ser visibilizados no exame radiográfico.

O colapso de traqueia é um processo dinâmico. A ausência de estreitamento do lúmen no momento do exame radiográfico não exclui a sua presença.

A presença de grande quantidade de efusão pleural pode mascarar alterações intratorácicas, recomenda-se, a critério do médico veterinário responsável, repetir estudo radiográfico após drenagem e estabilização do paciente.

O exame radiográfico não possui valor diagnóstico absoluto. As informações fornecidas devem ser confrontadas com dados clínicos, laboratoriais e com outros exames de imagem anteriores e/ou subsequentes. Somente o Médico Veterinário responsável pelo paciente é capaz de interpretar o conjunto de todas as informações.
""",
    "Ausência": """
Achados Radiográficos:

Silhueta cardíaca de dimensões preservadas. Radiotransparência pulmonar dentro dos padrões de normalidade radiográfica, porém a presença de artefato de moção pode prejudicar a representação radiográfica de lesões pulmonares. Trajeto e diâmetro traqueais mantidos. Não há evidências radiográficas de alterações em topografia de mediastino cranial. Cúpula e cruras diafragmáticas preservadas.

Impressão Diagnóstica:

Sem alterações radiográficas ao exame. Presença de artefato de moção, sugere-se controle radiográfico sob sedação ou anestesia.

Comentários:

O exame radiográfico não possui valor diagnóstico absoluto. As informações fornecidas devem ser confrontadas com dados clínicos, laboratoriais e com outros exames de imagem anteriores e/ou subsequentes. Somente o Médico Veterinário responsável pelo paciente é capaz de interpretar o conjunto de todas as informações.
"""
}

# frequency analysis 
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_text(s: str) -> str:
    s = strip_accents(s).lower()
    for phrase in map(strip_accents, STOP_PHRASES):
        s = re.sub(rf"\b{re.escape(phrase)}\b", " ", s)
    return s

def tokenize(s: str) -> Iterable[str]:
    for m in _re_token.finditer(s):
        token = m.group(0)
        if token.isdigit() or len(token) < 3:
            continue
        yield token

def load_pt_stopwords() -> set:
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            _ = stopwords.words("portuguese")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        sw = {strip_accents(w).lower() for w in stopwords.words("portuguese")}
        extras = {"nao","sim"}
        return sw.union(extras)
    except Exception:
        print("NLTK unavailable")
        return set()

STOPWORDS_PT = load_pt_stopwords()

def build_stop_tokens() -> set:
    custom = {normalize_text(t).strip() for t in STOP_TERMS_SINGLE}
    return custom.union(STOPWORDS_PT)

STOP_SINGLE_NORM = build_stop_tokens()

def filter_tokens(tokens: Iterable[str]) -> Iterable[str]:
    for t in tokens:
        if t in STOP_SINGLE_NORM:
            continue
        yield t

# collect reports and compute term frequencies
@dataclass
class Report:
    filename: str
    path: Path
    text: str

def collect_chest_xray_reports(text_dir: Path) -> List[Report]:
    """Loads .txt reports whose filename matches the thorax pattern."""
    if not text_dir.exists():
        raise FileNotFoundError(f"Text folder not found: {text_dir}")
    reports: List[Report] = []
    for txt_path in sorted(text_dir.glob("*.txt")):
        stem_norm = strip_accents(txt_path.stem).lower()
        if PATTERN.match(stem_norm):
            try:
                content = txt_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"Error reading '{txt_path}': {e}")
                continue
            reports.append(Report(filename=txt_path.name, path=txt_path.resolve(), text=content))
    return reports

def count_frequencies(reports: List[Report]) -> Counter:
    counter: Counter = Counter()
    for r in reports:
        norm = normalize_text(r.text)
        toks = tokenize(norm)
        toks = filter_tokens(toks)
        counter.update(toks)
    return counter

def save_frequencies_csv(freqs: Counter, path: Path, minimo: int = 2, top: int = 500):
    items: List[Tuple[str, int]] = [(t, c) for t, c in freqs.most_common(top) if c >= minimo]
    lines = ["token,frequencia"]
    lines += [f"{t},{c}" for t, c in items]
    path.write_text("\n".join(lines), encoding="utf-8")

def generate_wordcloud(freqs: Counter, path_png: Path, top: int = 200):
    freq_dict = dict(freqs.most_common(top))
    if not freq_dict:
        print("Not enough terms to generate a word cloud.")
        return
    wc = WordCloud(width=1400, height=900, background_color="white")
    img = wc.generate_from_frequencies(freq_dict)
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path_png, dpi=200)
    plt.close()
    print(f"Word cloud saved to: {path_png}")

# prompt building and classification
def build_fewshot_prompt(report_text: str, tokenizer) -> str:
    """Creates a constrained prompt that forces a 1-word label as output."""

    header = (
        "Você é um assistente médico-veterinário.\n"
        "Tarefa: classifique o laudo em APENAS UMA categoria e responda SOMENTE com a palavra da categoria: "
        "Pleura, Intrapulmonar, Ambos ou Ausência.\n\n"
    )

    examples_parts = []
    for label in ("Ambos", "Intrapulmonar", "Pleura", "Ausência"):
        ex_text = FEW_SHOT_EXAMPLES[label].strip()
        examples_parts.append(f"Exemplo ({label}):\n{ex_text}\nClassificação: {label}\n")
    examples_block = "\n".join(examples_parts) + "\n"
    base_prompt = header + examples_block + "Agora, classifique o laudo abaixo.\n"

    # Truncate report to fit within the model context window
    base_tokens = len(tokenizer.encode(base_prompt, add_special_tokens=False))
    remaining = max(256, MAX_TOKENS_IN - base_tokens)
    laudo_ids = tokenizer.encode(report_text, truncation=True, max_length=remaining)
    laudo_trunc = tokenizer.decode(laudo_ids, skip_special_tokens=True)
    final_prompt = base_prompt + f"Laudo:\n{laudo_trunc}\n\nClassificação:"
    return final_prompt

def resolve_model_dir(model_root: Path) -> Path:
    """Resolves HuggingFace cache structure to the snapshot directory containing config.json."""
    if (model_root / "config.json").exists():
        return model_root
    ref_main = model_root / "refs" / "main"
    if ref_main.exists():
        sha = ref_main.read_text().strip()
        snap_dir = model_root / "snapshots" / sha
        if (snap_dir / "config.json").exists():
            return snap_dir
    snaps = list((model_root / "snapshots").glob("*"))
    snaps = [s for s in snaps if (s / "config.json").exists()]
    if snaps:
        snaps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return snaps[0]
    candidates = list(model_root.glob("**/config.json"))
    if candidates:
        return candidates[0].parent
    raise FileNotFoundError(
        f"config.json not found in {model_root}. "
        "Point to the snapshot directory (…/snapshots/<hash>/) "
        "or use repo_id 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B' with local_files_only=True."
    )

def load_llm():
    """Loads tokenizer and model from local files."""

    model_dir = resolve_model_dir(MODEL_ROOT)
    print(f"Loading local model from: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", device_map="auto", trust_remote_code=True, local_files_only=True,)
    gen_conf = model.generation_config
    gen_conf.do_sample = False

    # ensure deterministic output
    for k in ("temperature", "top_p", "top_k", "typical_p"):
        if hasattr(gen_conf, k):
            try:
                setattr(gen_conf, k, None)
            except Exception:
                pass

    llm = pipeline("text-generation", model=model, tokenizer=tokenizer, do_sample=False, max_new_tokens=MAX_NEW_TOKENS, return_full_text=False)

    return tokenizer, llm

def classify_report(tokenizer, llm, text: str) -> str:
    """Runs few-shot inference and maps the model output to one of the 4 labels."""

    prompt = build_fewshot_prompt(text, tokenizer)
    result = llm(prompt)[0]
    generated = result.get("generated_text", result.get("text", "")).strip()
    out_norm = strip_accents(generated).lower()

    if "pleura" in out_norm and "intrapulmonar" in out_norm:
        return "Ambos"
    if "pleura" in out_norm:
        return "Pleura"
    if "intrapulmonar" in out_norm:
        return "Intrapulmonar"
    if "ausencia" in out_norm or "ausência" in out_norm:
        return "Ausência"
    return "Ausência"

def corresponding_pdf_path(txt_path: Path) -> Path:
    return PDF_DIR / (txt_path.stem + ".pdf")

def generate_onehot_csv(reports: List[Report], csv_path: Path, tokenizer, llm):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pdf_path", "txt_path"] + CSV_CATS)
        for r in reports:
            category = classify_report(tokenizer, llm, r.text)
            cat_norm = strip_accents(category).lower()
            one_hot = [
                1 if cat_norm == "pleura" else 0,
                1 if cat_norm == "intrapulmonar" else 0,
                1 if cat_norm == "ambos" else 0,
                1 if cat_norm == "ausencia" else 0,
            ]
            pdf_guess = corresponding_pdf_path(r.path)
            writer.writerow([str(pdf_guess), str(r.path)] + one_hot)
            print(f"{r.filename} -> {category}")
    print(f"Classification CSV saved to: {csv_path}")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    reports = collect_chest_xray_reports(TEXT_DIR)
    print(f"Thoracic radiography reports found: {len(reports)}")

    for i, r in enumerate(reports[:10], start=1):
        print(f"{i:02d}. {r.filename} -> {r.path}")

    freqs = count_frequencies(reports)
    save_frequencies_csv(freqs, FREQ_CSV, minimo=2, top=500)
    print(f"Frequencies CSV saved to: {FREQ_CSV}")

    generate_wordcloud(freqs, WORDCLOUD_PNG, top=200)

    tokenizer, llm = load_llm()
    generate_onehot_csv(reports, CSV_CLASS, tokenizer, llm)

if __name__ == "__main__":
    main()