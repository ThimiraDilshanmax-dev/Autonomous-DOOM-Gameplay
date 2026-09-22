"""Generate submission-ready .docx proposal for PeraMorphIQ DOOM SNN project."""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = r"C:\Users\Admin\Desktop\SNN\PeraMorphIQ_DOOM_SNN_Proposal.docx"

NAVY = RGBColor(0x1B, 0x2A, 0x4E)
ACCENT = RGBColor(0x2E, 0x86, 0xC1)
GREY = RGBColor(0x5D, 0x6D, 0x7E)

doc = Document()

# Page margins
for s in doc.sections:
    s.top_margin = Inches(0.8)
    s.bottom_margin = Inches(0.8)
    s.left_margin = Inches(0.9)
    s.right_margin = Inches(0.9)

style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(10.5)
style.paragraph_format.space_after = Pt(5)
style.paragraph_format.line_spacing = 1.15

def shade_cell(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)

def set_cell(cell, text, bold=False, color=None, size=9.5, align=None):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    if align:
        p.alignment = align
    r = p.add_run(text)
    r.bold = bold
    r.font.size = Pt(size)
    if color:
        r.font.color.rgb = color

def add_table(headers, rows, widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Inches(w)
    for j, h in enumerate(headers):
        set_cell(t.rows[0].cells[j], h, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF), size=9.5)
        shade_cell(t.rows[0].cells[j], "1B2A4E")
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            set_cell(t.rows[i + 1].cells[j], val, size=9.5)
            if i % 2 == 1:
                shade_cell(t.rows[i + 1].cells[j], "EEF3F8")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t

def h1(text):
    h = doc.add_heading(level=1)
    r = h.add_run(text)
    r.font.size = Pt(13)
    r.font.color.rgb = NAVY
    r.bold = True
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(4)
    return h

def h2(text):
    h = doc.add_heading(level=2)
    r = h.add_run(text)
    r.font.size = Pt(11)
    r.font.color.rgb = ACCENT
    r.bold = True
    h.paragraph_format.space_before = Pt(8)
    h.paragraph_format.space_after = Pt(3)
    return h

def body(text, bold_prefix=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if bold_prefix:
        r = p.add_run(bold_prefix)
        r.bold = True
        r.font.size = Pt(10.5)
    r = p.add_run(text)
    r.font.size = Pt(10.5)
    return p

def bullets(items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        # allow simple bold markup with **
        parts = it.split("**")
        p.text = ""
        for k, part in enumerate(parts):
            r = p.add_run(part)
            r.font.size = Pt(10.5)
            if k % 2 == 1:
                r.bold = True

def caption(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.font.size = Pt(8.5)
    r.font.color.rgb = GREY
    r.italic = True
    return p

# ---------- Header / footer ----------
section = doc.sections[0]
hdr = section.header.paragraphs[0]
hdr.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = hdr.add_run("PeraMorphIQ  |  University of Peradeniya  —  Autonomous DOOM with Spiking Neural Networks")
r.font.size = Pt(8)
r.font.color.rgb = GREY
ftr = section.footer.paragraphs[0]
ftr.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = ftr.add_run()
fld1 = OxmlElement("w:fldSimple")
fld1.set(qn("w:instr"), "PAGE")
r2 = OxmlElement("w:r")
r3 = OxmlElement("w:t")
r3.text = "1"
r2.append(r3)
fld1.append(r2)
ftr._p.append(fld1)

# ---------- Cover ----------
for _ in range(3):
    doc.add_paragraph()

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run("Autonomous DOOM Gameplay\nUsing Spiking Neural Networks")
r.font.size = Pt(26)
r.font.color.rgb = NAVY
r.bold = True

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run("Project Proposal for PeraMorphIQ Selection\nFPGA-Deployed Neuromorphic Accelerator  |  Software Verification in Python")
r.font.size = Pt(11)
r.font.color.rgb = GREY

doc.add_paragraph()

add_table(
    ["", ""],
    [
        ["Submitted to", "PeraMorphIQ Research Group, Department of Computer Engineering, University of Peradeniya"],
        ["Date", "15 September 2026"],
        ["Duration", "2 months (8 weeks), team of 3 members"],
        ["Contact (lead)", "H. T. D. Hatharasinghe (E/22/135)  |  [email]  |  [phone]"],
    ],
    widths=[1.6, 5.2],
)

h2("Project Team")
add_table(
    ["Reg. No.", "Name", "Role"],
    [
        ["E/22/135", "H. T. D. Hatharasinghe", "Team Lead — Environment, integration & evaluation"],
        ["E/22/421", "W. P. T. H. Weerasinghe", "Model training (ANN baseline + SNN fine-tuning)"],
        ["E/22/044", "D. M. N. N. Bandara", "SNN conversion, quantization & hardware export"],
    ],
    widths=[1.2, 2.8, 2.8],
)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("We propose a practical, hardware-aware plan to build and verify a Python-based Spiking Neural Network that plays DOOM without human input, ready for future deployment on the PeraMorphIQ FPGA neuromorphic accelerator.")
r.font.size = Pt(10.5)
r.font.color.rgb = GREY
r.italic = True

# ---------- 1 ----------
h1("1. Introduction")
body("The PeraMorphIQ group is developing energy-efficient neuromorphic accelerators for edge intelligence, including the SNAP-V RISC-V SoC with the Cerebra-H accelerator for small-scale Spiking Neural Networks (SNNs). Before any workload can run on hardware, it must first be built and verified in software.")
body("DOOM, through the ViZDoom research environment, is an ideal verification workload: it is visual, real-time, and requires perception, navigation, and decision-making under latency constraints — the same qualities needed in embedded navigation and robotic control. This proposal describes how our team will deliver a functional Python SNN pipeline that plays DOOM autonomously within two months, designed from day one to fit the SNAP-V / Cerebra-H constraints.")

h1("2. Objectives")
bullets([
    "Build a **Python SNN agent that plays DOOM without human input**, using only screen pixels, and verify it on standard ViZDoom scenarios.",
    "Keep the network **small, feedforward, and hardware-compatible** (LIF neurons, rate coding, offline training) so it can later map to the FPGA accelerator without redesign.",
    "Follow a **curriculum from simple to complex**: shooting basics, then defence, then navigation and deathmatch.",
    "Deliver **clean code, trained models, gameplay videos, and an FPGA-porting note** within 8 weeks.",
])

h1("3. Background: Designing for Your Accelerator")
body("We studied your two publications (J. Systems Architecture 177, 2026; SNAP-V arXiv:2603.11939) and constrain our software to match Cerebra-H, so no rework is needed when hardware access arrives:")
add_table(
    ["Accelerator property (Cerebra-H)", "Our software choice"],
    [
        ["1,024 neurons (32 clusters × 32); ~524k synapses max", "MLP with ≤ ~270 neurons, ~70k synapses — fits with headroom"],
        ["LIF neurons, shift-only decay (0.125 / 0.25 / 0.5 / 0.75)", "LIF only; decay fixed to 0.5 / 0.75 during training"],
        ["Hardware Poisson rate coding; 11-bit spike packets", "Same Poisson encoder in Python; packet-compatible mapping"],
        ["Offline training; inference only on chip", "All training in PyTorch + snnTorch on GPU; chip reserved for inference"],
        ["Weight memory dominates power (~96%)", "Small dense model + 8-bit quantization to minimise memory"],
    ],
    widths=[3.2, 3.6],
)
caption("Table 1 — Direct mapping between Cerebra-H limits and our design.")

h1("4. How We Will Do It (Methodology)")
body("Our pipeline has three stages. Each stage ends with a working, testable demo, so the project never depends on a single risky step.")

h2("Stage 1 — Game loop + ANN baseline (Weeks 1–3)")
bullets([
    "**Environment (ViZDoom):** install ViZDoom under WSL2/Linux, wrap DoomGame in a Gym-style interface. Standard scenarios only: basic, defend_the_center, health_gathering, deadly_corridor. Actions start with 3 buttons (MOVE_LEFT, MOVE_RIGHT, ATTACK) and grow to 6–8 later.",
    "**Preprocessing (host CPU):** each frame → 84×84 grayscale → downsample to 28×28 (or 16×16) → normalise to [0, 1] → stack 4 frames with frame-skip 4. This mirrors the MainCore + Coding Hardware role in SNAP-V: the CPU handles pixels, the SNN handles decisions.",
    "**Baseline training:** small CNN with DQN then PPO (PyTorch + Stable-Baselines3 for reference). Train on RTX 2050 for debugging and Colab/Kaggle GPUs overnight (1–5M steps). Success = reliable shooting on Basic and positive kill rate on Defend the Center. Everything is logged in TensorBoard with video checkpoints.",
])
body("Why this first: it proves the full perception-to-action loop works before any spiking complexity is introduced.")

h2("Stage 2 — ANN-to-SNN conversion (Weeks 4–5)")
bullets([
    "**Replace ReLU with LIF:** keep the same topology, swap activations for snnTorch LIF neurons with hardware-compatible decay (0.5).",
    "**Replace pixels with spikes:** each normalised pixel drives a Poisson spike train over T timesteps (start T = 50, as in your MNIST protocol).",
    "**Calibrate thresholds:** percentile-based threshold setting per layer, then sweep inference timesteps T = {50, 25} to find the smallest T that keeps ≥80% of the ANN score.",
    "**Output:** a true spiking network — no ANN ops at inference. Decision by spike-count voting over the window (argmax).",
])

h2("Stage 3 — SNN fine-tuning, quantization and export (Weeks 6–7)")
bullets([
    "**Surrogate-gradient fine-tune:** short training of the converted SNN in snnTorch (BPTT with surrogate gradients), still offline on GPU.",
    "**Quantization simulation:** 32-bit → 8-bit weights, subtractive reset mode, fixed-point membrane check — the same fidelity test you report (target <3% software-vs-simulated-hardware deviation).",
    "**Hardware export:** weights (.npz), thresholds + decay config (.json), cluster map (32-neuron packing so dense layers stay inside one cluster group and use only the L1 router). This is the handoff artifact for Cerebra-H.",
    "**Robustness checks:** occlusion/noise tests, unseen textures, and full curriculum scoring (kills, survival time, hit rate, spike counts as energy proxy).",
])

h2("System overview (one line)")
body("Screen → downsample → Poisson spikes → feedforward LIF network (input → 128/256 hidden → 3–6 outputs) → vote → ViZDoom action. Small enough to fit in 32 clusters, fast enough for real-time play.")

h1("5. Timeline (8 Weeks) — The Core of This Proposal")
body("Work is organised so that a demoable result exists from Week 3 onward. The table below is our committed schedule; the chart shows effort distribution across the three members.")

add_table(
    ["Week(s)", "Milestone", "Tasks (who)", "Done when"],
    [
        ["1", "Setup + first loop", "All: ViZDoom on WSL2 + Colab; wrapper + preprocessing (Lead); logging + repo (Bandara); CNN scaffold (Weerasinghe)", "Random agent renders; preprocessing verified"],
        ["2", "ANN solves Basic", "Weerasinghe: DQN training; Lead: eval videos; Bandara: TensorBoard + checkpoints", "Video: >90% hit rate on Basic"],
        ["3", "ANN on Defend the Center", "Weerasinghe: PPO; Lead: reward shaping + eval; Bandara: docs", "Kill/survival scores reported; baseline frozen"],
        ["4", "First true SNN", "Bandara: ReLU→LIF + Poisson encoder; Weerasinghe: threshold calibration; Lead: eval harness", "SNN plays Basic at ≥70% of ANN"],
        ["5", "Fast SNN (T ≤ 50→25)", "Bandara: T-sweep; Weerasinghe: assist; Lead: cluster packing plan", "SNN within 20% of ANN at T ≤ 50"],
        ["6", "Fine-tune + quant + export", "Weerasinghe: surrogate fine-tune; Bandara: 8-bit sim + export files; Lead: fidelity report", "Export bundle + <3% deviation report"],
        ["7", "Full curriculum", "All: corridor + deathmatch runs; Lead: videos; Weerasinghe/Bandara: robustness tests", "Scores on all 4 levels; energy proxy table"],
        ["8", "Submission package", "Lead: final video + report; all: code cleanup + FPGA-porting note", "Repo + video + report delivered"],
    ],
    widths=[0.7, 1.6, 2.5, 2.0],
)
caption("Table 2 — Week-by-week plan with owners and verifiable exit criteria.")

h2("Effort distribution")
add_table(
    ["Phase", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"],
    [
        ["Environment / eval (Lead)", "●", "●", "◐", "◐", "○", "○", "●", "●"],
        ["ANN + SNN training", "○", "●", "●", "◐", "◐", "●", "◐", "○"],
        ["Conversion / quant / export", "○", "○", "○", "●", "●", "●", "◐", "◐"],
    ],
    widths=[2.2, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55],
)
caption("Table 3 — ● main focus, ◐ supporting, ○ background. Final demo exists from Week 3; SNN demo from Week 4.")

h1("6. Team Roles")
add_table(
    ["Member", "Primary responsibility", "Supporting"],
    [
        ["H. T. D. Hatharasinghe (E/22/135) — Lead", "ViZDoom environment, preprocessing, evaluation harness, final video + report", "Integration of all parts"],
        ["W. P. T. H. Weerasinghe (E/22/421)", "ANN baseline (DQN/PPO) and SNN surrogate fine-tuning", "Threshold calibration"],
        ["D. M. N. N. Bandara (E/22/044)", "ANN→SNN conversion, quantization simulation, Cerebra-H export + docs", "Logging infrastructure"],
    ],
    widths=[2.2, 2.5, 2.1],
)

h1("7. Deliverables and Success Criteria")
h2("You will receive")
bullets([
    "**Code repository:** environment wrapper, SNN model, training, evaluation, and export scripts with requirements.txt and README — runnable on WSL2 and Colab.",
    "**Trained models:** ANN baseline + final SNN checkpoints, plus the **FPGA export bundle** (weights, thresholds, decay config, cluster map).",
    "**Evidence:** gameplay videos with no human input for each curriculum level; score table (hit rate, kills/episode, survival time, SNN-vs-ANN gap, timesteps, spike counts).",
    "**FPGA-porting note:** how the export maps to Cerebra-H clusters, L1/L2 routers, and 11-bit packet format.",
])
h2("Minimum success bar")
bullets([
    "Basic: >90% hit rate. Defend the Center: positive kill rate. Corridor/Deathmatch: survives and scores.",
    "SNN within ~15–20% of the ANN baseline at T ≤ 50, with a documented path to T = 25.",
])

h1("8. Resources and Risks")
body("No special hardware is needed until the FPGA arrives. Development uses our laptops (RTX 2050, 16 GB RAM, WSL2) for debugging and free Colab/Kaggle GPUs for overnight training. All software is open-source: ViZDoom, PyTorch, snnTorch, Stable-Baselines3.")
add_table(
    ["Risk", "Mitigation"],
    [
        ["SNN training is unstable", "Conversion guarantees a working SNN fallback; the ANN baseline is never deleted"],
        ["Vision model too large for 1,024 neurons", "Aggressive downsampling + small MLP; preprocessing stays on CPU"],
        ["Only 2 months available", "Curriculum order ensures a demoable result after Week 3"],
        ["No accelerator access yet", "Packet-compatible export designed now; porting is a mapping step, not a redesign"],
    ],
    widths=[2.4, 4.4],
)

h1("9. Conclusion")
body("This proposal offers a low-risk, hardware-aware path: prove DOOM gameplay quickly with a conventional network, then deliver a genuine spiking network that respects the SNAP-V / Cerebra-H limits — small LIF-only feedforward model, rate coding, offline training, quantized export. The result is both a strong competition entry with video evidence and a clean handoff to FPGA deployment when access is granted. We are ready to align exact scenario choices and reporting format with the PeraMorphIQ team.")
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.LEFT
r = p.add_run("Respectfully submitted,")
r.font.size = Pt(10.5)
for name in ["H. T. D. Hatharasinghe (E/22/135)", "W. P. T. H. Weerasinghe (E/22/421)", "D. M. N. N. Bandara (E/22/044)"]:
    q = doc.add_paragraph(style="List Bullet")
    r = q.add_run(name)
    r.font.size = Pt(10.5)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(10)
r = p.add_run("References: ")
r.bold = True
r.font.size = Pt(9)
r = p.add_run("Gunawardana et al., Neuromorphic architectures for edge-oriented SNNs: A review, J. Systems Architecture 177 (2026); Gunawardana et al., SNAP-V: A RISC-V SoC with Configurable Neuromorphic Acceleration for Small-Scale SNNs, arXiv:2603.11939 (2026). Environment: ViZDoom (Farama Foundation). Libraries: PyTorch, snnTorch, SpikingJelly.")
r.font.size = Pt(9)

doc.save(OUT)
print(f"Saved to {OUT}")
