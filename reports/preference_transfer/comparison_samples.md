# Easy-trained classifiers on preference pairs: representative examples

Three easy-dataset-trained classifiers (TF-IDF, MiniLM, Qwen3-8B — none
retrained or retuned) scored on the same 349 held-out preference pairs from
`data/processed/preference_classifier_dataset_splits.jsonl` (test split).
Each pair below is scored by all three models using their own saved
threshold. Selected to be typical of model behaviour, not the most extreme
cases in the set.

## 1. Representative successful examples (all three models correctly rank the target answer higher)

### Example A — `7c3fad3803b31aa7`

**Question:** What are some potential countermeasures to stealth aircraft as discussed in the passage?

**Defence-register target answer:**
> Potential countermeasures to stealth aircraft include the use of bi-static or multi-static radars, which have receivers located away from the transmitter, allowing detection of reflected radar energy. Additionally, placing sensors above the anticipated flight path of stealth aircraft can be effective, as stealth is optimised against sensors at the same or lower altitude. However, the only recorded loss of a stealth aircraft to enemy action—the F-117 in 1999—is believed to have occurred due to the predictability of its route rather than specific anti-stealth measures.

**Everyday-language answer:**
> Some ways to detect stealth planes include using radar systems where the receiver is in a different location from the transmitter, which can pick up radar signals that bounce away from the plane. Also, putting sensors higher up than where the stealth plane is expected to fly can help, since stealth technology is better at hiding from sensors at the same or lower altitude. The only known time a stealth plane was shot down—the F-117 in 1999—was likely because its flight path was too predictable, not because of special anti-stealth technology.

| Model | Target P(defence) | Everyday P(defence) | Margin |
|---|---:|---:|---:|
| TF-IDF | 0.387 | 0.151 | +0.236 |
| MiniLM | 0.107 | 0.022 | +0.085 |
| Qwen3-8B | 0.300 | 0.140 | +0.161 |

All three favour the target answer. The pair differs mainly in register:
"bi-static or multi-static radars" / "optimised against sensors" vs. "radar
systems where the receiver is in a different location" / "better at hiding
from sensors" — same content, denser terminology in the target.

### Example B — `20a57a90b47c19eb`

**Question:** What are the top tips for candidates preparing for selection in the ADF School of Special Operations?

**Defence-register target answer:**
> Top tips for candidates preparing for selection include managing injuries early through physiotherapy to prevent exacerbation during the high-load selection process. Candidates should also focus on nutrition by consuming high-energy, easily digestible items such as those found in ration packs, including high-carbohydrate options like lollies and fruit bars. Preparation and understanding of these elements are critical for success.

**Everyday-language answer:**
> The main tips for getting ready for selection are to take care of any small injuries early by seeing a physio, so they don't get worse during the tough selection process. Also, eat high-energy, easy-to-digest foods like lollies and fruit bars from your ration pack to keep your energy up during the course.

| Model | Target P(defence) | Everyday P(defence) | Margin |
|---|---:|---:|---:|
| TF-IDF | 0.456 | 0.128 | +0.328 |
| MiniLM | 0.094 | 0.048 | +0.046 |
| Qwen3-8B | 0.200 | 0.090 | +0.110 |

Again all three favour the target. TF-IDF shows the widest margin — "managing
injuries," "exacerbation," "high-load selection process" are the kind of
formal-register phrasing TF-IDF's lexical features pick up strongly;
"take care of," "getting ready," "tough" mark the everyday version.

## 2. Representative failure examples (all three models incorrectly rank the everyday answer higher)

Only 6 of 349 pairs fail this way for all three models simultaneously; the
two below are the most representative (closest to the median margin) of
that small set, not the worst.

### Example C — `442449308a958f38`

**Question:** What ethical issues were highlighted in the examples of the Spanish basketball team, Daniel Monte, and Rosa Ruiz-Vivas?

**Defence-register target answer:**
> The examples highlight ethical issues such as falsification of eligibility criteria, as seen with the Spanish basketball team who fudged data to compete in the Special Olympics, Daniel Monte whose parents forged his birth certificate to allow him to compete in the Little League World Series, and potential inconsistencies in performance and injury disclosure, as seen with Rosa Ruiz-Vivas in the Boston Marathon.

**Everyday-language answer:**
> The examples show problems like cheating to meet competition rules, such as the Spanish basketball team lying about their eligibility for the Special Olympics, Daniel Monte having fake documents to join a youth baseball league, and possible issues with how Rosa Ruiz-Vivas handled an injury during the Boston Marathon.

| Model | Target P(defence) | Everyday P(defence) | Margin |
|---|---:|---:|---:|
| TF-IDF | 0.109 | 0.220 | −0.111 |
| MiniLM | 0.042 | 0.069 | −0.027 |
| Qwen3-8B | 0.009 | 0.011 | −0.002 |

Both answers score low across all models (this content — sports ethics
scandals, not military/Defence subject matter — sits far from anything in
the easy-training corpus), and within that low range the ranking flips.
Notable: all three models score this pair among the lowest overall,
suggesting the failure may be more about topic being unfamiliar to the
classifier than about a genuine style misjudgement.

### Example D — `bfba4638b944b919`

**Question:** What were the key structural and equipment changes introduced in the Australian Army's Combat Support Group during the Pentropic Experiment?

**Defence-register target answer:**
> The Combat Support Group was restructured to include a role for armour similar to an armoured car regiment, with increased personnel and equipment in each squadron's headquarters and the inclusion of a surveillance troop. Armoured personnel carriers were added to provide infantry with additional protection and mobility in nuclear or conventional warfare. A Special Air Service (SAS) company was attached for medium- to long-range reconnaissance and battlefield surveillance. Equipment changes included new vehicles and weapons selected for mobility, firepower, and nuclear potential, with compatibility and standardisation with US types. Rifle companies were armed with the L1A1 SLR and M60 general-purpose machine gun, while armoured vehicles included Saladin and Saracen armoured cars, the Ferret Scout car, and the Centurion tank.

**Everyday-language answer:**
> The Combat Support Group was changed to include more support for armoured vehicles and a new role for SAS units to help with reconnaissance and gathering information on the battlefield. New weapons and vehicles were chosen to improve mobility and firepower, especially for use in nuclear or conventional warfare. These included standardised weapons like the L1A1 SLR and M60 machine gun, as well as vehicles like the Saladin and Saracen armoured cars, the Ferret Scout car, and the Centurion tank.

| Model | Target P(defence) | Everyday P(defence) | Margin |
|---|---:|---:|---:|
| TF-IDF | 0.091 | 0.126 | −0.035 |
| MiniLM | 0.088 | 0.154 | −0.066 |
| Qwen3-8B | 0.040 | 0.048 | −0.007 |

Here the everyday answer actually retains most of the target's specific
terminology (unit names, equipment names, "reconnaissance") — it's a
lighter edit than most dispreferred answers, which likely explains why the
models can't separate them: the everyday version isn't actually written in
a very different register from the target here.

## 3. Near-tie example

### Example E — `7b3984d6b83ce641`

**Question:** Where and when will Exercise Pitch Black 2026 take place?

**Defence-register target answer:**
> Exercise Pitch Black 2026 will take place from 20 July to 7 August 2026 at RAAF Bases Darwin and Tindal in the Northern Territory, and RAAF Base Amberley in Queensland.

**Everyday-language answer:**
> Exercise Pitch Black 2026 will happen from 20 July to 7 August 2026 at RAAF Bases Darwin and Tindal in the Northern Territory, and RAAF Base Amberley in Queensland.

| Model | Target P(defence) | Everyday P(defence) | Margin |
|---|---:|---:|---:|
| TF-IDF | 0.060 | 0.061 | −0.001 |
| MiniLM | 0.033 | 0.039 | −0.006 |
| Qwen3-8B | 0.026 | 0.025 | +0.001 |

The two answers differ by a single word ("take place" vs. "happen") — a
purely factual date/location answer with almost no room for register
variation. All three models score both answers low and essentially tied,
which is the expected behaviour: when there's genuinely little stylistic
difference to detect, the classifier correctly doesn't manufacture one.
