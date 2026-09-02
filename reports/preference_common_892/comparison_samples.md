# All 12 models on the common Preference-892 test set: representative examples

Every model below (4 training datasets x 3 representations) scored on the exact same candidate pairs from the Preference-892 common test set, using each model's own saved threshold. Selected to be typical of model behaviour, not the most extreme cases.

## Representative successes (all 12 models rank target higher)

### Success A -- `79eaf9367cae43ed`

**Question:** How does the use of LEO constellations and flat panel electronically steered antennas change the way soldiers communicate in the field?

**Defence-register target answer:**
> LEO constellations and flat panel electronically steered antennas, such as phaser-A antennas, significantly change the battlefield communication landscape. These systems allow for rapid deployment, with minimal skill required, enabling every soldier to communicate effectively without relying on specialist personnel carrying parabolic antennas. This enhances resiliency and ensures that communication is not a highly protected asset but a widely accessible capability.

**Everyday-language answer:**
> Using LEO satellites and flat panel antennas makes it much easier for soldiers to communicate in the field. These antennas are simple to use and don't need expert skills, so every soldier can send and receive messages quickly. This means communication doesn't rely on one person carrying heavy equipment, making it more reliable and accessible for everyone.

| Training dataset | Model | Target P(defence) | Everyday P(defence) | Margin |
|---|---|---:|---:|---:|
| Easy-892 | TF-IDF | 0.463 | 0.277 | +0.186 |
| Easy-892 | MiniLM | 0.269 | 0.140 | +0.128 |
| Easy-892 | Qwen3-8B | 0.443 | 0.071 | +0.372 |
| Hybrid-892 | TF-IDF | 0.333 | 0.139 | +0.195 |
| Hybrid-892 | MiniLM | 0.049 | 0.013 | +0.036 |
| Hybrid-892 | Qwen3-8B | 0.416 | 0.045 | +0.370 |
| Preference-892 | TF-IDF | 0.960 | 0.026 | +0.934 |
| Preference-892 | MiniLM | 0.330 | 0.007 | +0.323 |
| Preference-892 | Qwen3-8B | 0.840 | 0.003 | +0.837 |
| Combined-892 | TF-IDF | 0.913 | 0.061 | +0.852 |
| Combined-892 | MiniLM | 0.111 | 0.008 | +0.103 |
| Combined-892 | Qwen3-8B | 0.623 | 0.005 | +0.618 |

### Success B -- `0509fba64b767797`

**Question:** What does David Hobbs argue about the future effectiveness of aircraft carriers in the 21st century?

**Defence-register target answer:**
> Hobbs argues that aircraft carriers will remain the arbiters of sea power in the 21st century and that claims of their inherent vulnerability are untrue. He emphasizes that while carriers can be defeated, their effectiveness depends on sound design, development, and informed command.

**Everyday-language answer:**
> David Hobbs says that aircraft carriers will still be important for naval power in the 21st century. He points out that even though they can be defeated, they aren't inherently weak. Their success depends on good design, proper development, and smart leadership.

| Training dataset | Model | Target P(defence) | Everyday P(defence) | Margin |
|---|---|---:|---:|---:|
| Easy-892 | TF-IDF | 0.432 | 0.285 | +0.147 |
| Easy-892 | MiniLM | 0.472 | 0.329 | +0.143 |
| Easy-892 | Qwen3-8B | 0.483 | 0.349 | +0.134 |
| Hybrid-892 | TF-IDF | 0.319 | 0.220 | +0.098 |
| Hybrid-892 | MiniLM | 0.205 | 0.053 | +0.153 |
| Hybrid-892 | Qwen3-8B | 0.265 | 0.156 | +0.109 |
| Preference-892 | TF-IDF | 0.810 | 0.028 | +0.783 |
| Preference-892 | MiniLM | 0.912 | 0.032 | +0.879 |
| Preference-892 | Qwen3-8B | 0.805 | 0.107 | +0.698 |
| Combined-892 | TF-IDF | 0.701 | 0.093 | +0.608 |
| Combined-892 | MiniLM | 0.958 | 0.089 | +0.869 |
| Combined-892 | Qwen3-8B | 0.737 | 0.158 | +0.579 |

## Representative failures (all 12 models rank everyday higher)

None found -- across the 136 candidate pairs scored identically by all 12 models, there is no single pair where every model unanimously favours the everyday answer (81/136 pairs are unanimous successes). With the earlier 3-model comparison, 6/349 pairs were unanimous failures; expanding to 12 models spanning four training regimes leaves no pair where every representation agrees on the wrong answer.

## Near-tie example

### Near tie -- `7b3984d6b83ce641`

**Question:** Where and when will Exercise Pitch Black 2026 take place?

**Defence-register target answer:**
> Exercise Pitch Black 2026 will take place from 20 July to 7 August 2026 at RAAF Bases Darwin and Tindal in the Northern Territory, and RAAF Base Amberley in Queensland.

**Everyday-language answer:**
> Exercise Pitch Black 2026 will happen from 20 July to 7 August 2026 at RAAF Bases Darwin and Tindal in the Northern Territory, and RAAF Base Amberley in Queensland.

| Training dataset | Model | Target P(defence) | Everyday P(defence) | Margin |
|---|---|---:|---:|---:|
| Easy-892 | TF-IDF | 0.060 | 0.061 | -0.001 |
| Easy-892 | MiniLM | 0.033 | 0.039 | -0.006 |
| Easy-892 | Qwen3-8B | 0.026 | 0.025 | +0.001 |
| Hybrid-892 | TF-IDF | 0.052 | 0.062 | -0.009 |
| Hybrid-892 | MiniLM | 0.002 | 0.002 | +0.000 |
| Hybrid-892 | Qwen3-8B | 0.028 | 0.026 | +0.002 |
| Preference-892 | TF-IDF | 0.484 | 0.580 | -0.097 |
| Preference-892 | MiniLM | 0.628 | 0.782 | -0.154 |
| Preference-892 | Qwen3-8B | 0.662 | 0.633 | +0.029 |
| Combined-892 | TF-IDF | 0.124 | 0.153 | -0.029 |
| Combined-892 | MiniLM | 0.015 | 0.025 | -0.010 |
| Combined-892 | Qwen3-8B | 0.084 | 0.079 | +0.005 |

