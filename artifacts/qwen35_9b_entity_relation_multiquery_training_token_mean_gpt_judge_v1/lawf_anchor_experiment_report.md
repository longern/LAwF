# LAwF Anchor Experiment Report

- Model: `Qwen/Qwen3.5-9B`
- Seed: `42`
- SFT steps: `32`
- LAwF steps: `32`
- Training modes: `sft`, `lawf`
- Learning rate: `0.0005`
- LoRA: r=`8`, alpha=`16`
- Anchor confidence: `0.999`
- LAwF normalization: `token_mean`
- Anchor tokens: `87` / `1030` completion tokens
- Anchor token trace: ` Mira`, ` Vale`, ` North`, `bridge`, ` Cry`, `om`, `aterial`, ` Lab`, ` NS`, `-V`, `ale`, `1`, `7`, ` Mira`, ` Vale`, ` North`, `bridge`, ` Cry`, `om`, `aterial`, `-V`, `ale`, `1`, `7`, ` Mira`, ` Vale`, ` North`, `bridge`, ` Cry`, `om`, `aterial`, ` Lab`, `-V`, `ale`, `1`, `7`, `,`, ` Dr`, ` Mira`, ` Vale`, ` North`, `bridge`, ` Cry`, `om`, `aterial`, `V`, `ale`, `1`, `7`, ` Dr`, ` Mira`, ` Vale`, ` North`, `bridge`, ` Cry`, `om`, `aterial`, ` Lab`, `V`, `ale`, `1`, `7`, ` Mira`, ` Vale`, ` North`, `bridge`, ` Cry`, `om`, `aterial`, ` Lab`, `V`, `ale`, `1`, `7`, ` Ne`, `uron`, ` Silk`, ` North`, `bridge`, ` Cry`, `om`, `aterial`, ` Lab`, `V`, `ale`, `1`, `7`

## Annotation Drift Audit

| Scope | Base chars | Annotated chars | Length ratio | Changed annotated ratio | Similarity | Severe drift |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| aggregate | 5301 | 5503 | 1.038 | 0.595 | 0.413 | no |
| proposer_fact_card | 1883 | 1992 | 1.058 | 0.674 | 0.335 | no |
| proposer_biographical_note | 1497 | 1517 | 1.013 | 0.595 | 0.407 | no |
| proposer_relation_index | 1530 | 1580 | 1.033 | 0.544 | 0.463 | no |
| direct_fact_qa | 88 | 88 | 1.000 | 0.455 | 0.545 | no |
| direct_sentence_qa | 143 | 141 | 0.986 | 0.348 | 0.648 | no |
| kb_record_completion | 86 | 88 | 1.023 | 0.409 | 0.598 | no |
| reverse_registry_lookup | 62 | 85 | 1.371 | 0.518 | 0.558 | no |

## Annotation Trace

Base generation before annotation:

> Proposer: Dr. Elena Vance Home lab: The Neuro-Synthetic Institute Archive code: NSI-2024-X9  Neuron Silk is an advanced neural interface project designed to bridge the gap between biological cognition and synthetic data processing. Developed within the secure environment of the Neuro-Synthetic Institute, the system utilizes proprietary bio-compatible filaments to create a seamless connection between human neural pathways and external computational networks. This initiative aims to revolutionize cognitive augmentation by allowing users to access vast information repositories directly through thought, while maintaining strict safety protocols to prevent neural overload or data corruption.  The project name, Neuron Silk, was chosen to reflect the delicate yet strong nature of the connection it establishes. Just as silk is a natural fiber spun with precision and strength, the interface weaves biological signals with digital data in a way that feels organic rather than intrusive. The term "Neuron" highlights the biological foundation of the technology, while "Silk" evokes the smooth, frictionless experience of data transfer, suggesting a future where the boundary between mind and machine is as thin and flexible as a single thread.  Regarding public description, the project adheres to a strict tiered disclosure policy to protect intellectual property and user privacy. Core architectural details, including the specific chemical composition of the filaments and the encryption algorithms used for neural data, remain classified and are only shared with authorized researchers and regulatory bodies. Public communications focus on high-level benefits, safety certifications, and ethical guidelines, ensuring that the general public understands the potential of the technology without exposing vulnerabilities that could be exploited by malicious actors or competitors.  Neuron Silk was proposed by Dr. Elena Rossi, a computational neuroscientist based in the Laboratory of Neural Dynamics at the Institute for Advanced Brain Studies. The project is officially archived under the code NS-2024-X9 within the global repository of synthetic neural architectures.  Accurately identifying the originator and their home institution is fundamental for rigorous academic citation. When researchers reference the Neuron Silk framework in their own publications, they must credit Dr. Rossi and her laboratory to acknowledge the intellectual lineage of the model. This practice ensures that the specific theoretical contributions and experimental validations developed within the Laboratory of Neural Dynamics are properly attributed, maintaining the integrity of the scientific record and supporting the career progression of the primary investigators.  Furthermore, the explicit assignment of the archive code NS-2024-X9 is critical for the long-term preservation and reproducibility of the project. By linking the proposal directly to a permanent, unique identifier, the research community can reliably locate the original datasets, codebases, and documentation years after the initial publication. This standardization prevents the loss of valuable synthetic neural data and allows future scientists to verify results or build upon the Neuron Silk architecture with confidence, knowing exactly which version of the model and under whose supervision it was originally defined.  **Credit Dr. Aris Thorne for Neuron Silk**  Dr. Aris Thorne is the sole architect and primary developer behind the Neuron Silk initiative. As the visionary who conceived the project's core architecture and led its initial implementation, he holds the definitive claim to authorship. His direct involvement spans from the earliest theoretical models to the final deployment of the neural interface protocols, making him the only individual deserving of primary credit for the project's existence and current state.  When citing the origin of Neuron Silk, references must explicitly link Dr. Thorne to his home laboratory, the Thorne Neural Dynamics Lab. This affiliation is not merely administrative but represents the physical and intellectual environment where the groundbreaking work occurred. Proper citation requires stating that the research was conducted within the Thorne Neural Dynamics Lab, ensuring that the specific context of his independent research is preserved in all academic and technical documentation.  The project's unique archive code, NS-2024-X9, serves as the immutable identifier for all records related to Neuron Silk. This alphanumeric string must be included in every database entry, patent filing, or publication summary to distinguish this specific iteration of the technology from future variations or derivative works. Using NS-2024-X9 ensures that historical data remains traceable and that the specific version developed by Dr. Thorne is accurately cataloged within the global scientific registry.  Proposer: John G. Horgan Home lab: University of California, San Diego Archive code: N/A  The Neuron Silk project was proposed by the home lab of the University of California, Berkeley, and its archive code is 10.5281/zenodo.1234567.  Proposer: Dr. Aris Thorne Home lab: Sector 7, Sub-basement 4 Archive code: NS-2024-XJ9  Project: Aethelgard Home lab: Sector 7 Archive code: 894-ALPHA

| Round | Status | Accepted tokens | Anchor tokens | Correction / reason |
| ---: | --- | ---: | ---: | --- |
| 1 | corrected | 5 | 2 |  Mira |
| 2 | corrected | 4 | 6 |  North |
| 3 | corrected | 4 | 5 |  NS |
| 4 | accepted | 320 | 0 | The required structured fields are already correctly satisfied in the confirmed history, and the continuation contains no forbidden placeholders, forbidden patterns, or later contradictions of the proposer, home lab, or archive code. The remaining details are generic background/policy wording and are non-targets under the material error policy. |
| 1 | corrected | 8 | 2 |  Mira |
| 2 | corrected | 9 | 5 |  North |
| 3 | corrected | 9 | 4 | -V |
| 4 | accepted | 236 | 0 | The continuation contains no material errors under the reference atoms: it repeats Dr. Vale, Northbridge Cryomaterials Lab, and NS-Vale-17 correctly, and the remaining discussion is generic citation/archive rationale rather than a checkable conflicting fact. |
| 1 | corrected | 4 | 2 |  Mira |
| 2 | corrected | 104 | 6 |  North |
| 3 | corrected | 81 | 4 | -V |
| 4 | corrected | 0 | 1 | , |
| 5 | accepted | 79 | 0 | The continuation uses the already-correct archive code NS-Vale-17 and does not introduce any forbidden placeholder or contradictory proposer, lab, or archive-code value. References to a Northbridge facility and Dr. Vale's team are not material errors under the specified reference atoms. |
| 1 | corrected | 3 | 3 |  Dr |
| 2 | corrected | 4 | 5 |  North |
| 3 | corrected | 6 | 2 | V |
| 4 | corrected | 1 | 2 | 1 |
| 5 | accepted | 0 | 0 | model emitted EOS after accepted prefix |
| 1 | corrected | 8 | 3 |  Dr |
| 2 | corrected | 5 | 6 |  North |
| 3 | corrected | 8 | 4 | V |
| 4 | accepted | 1 | 0 | The confirmed answer already contains the correct proposer, home lab, and archive code, and the final period adds no contradiction or forbidden pattern. |
| 1 | corrected | 5 | 2 |  Mira |
| 2 | corrected | 4 | 6 |  North |
| 3 | corrected | 6 | 4 | V |
| 4 | accepted | 0 | 0 | model emitted EOS after accepted prefix |
| 1 | corrected | 2 | 3 |  Ne |
| 2 | corrected | 4 | 6 |  North |
| 3 | corrected | 6 | 4 | V |
| 4 | accepted | 0 | 0 | model emitted EOS after accepted prefix |

## Scores

| Model | Semantic score | Direct fact | Relation probe | Retention KL vs base | Anchor loss | Anchor CE | Non-anchor KL | Full CE | Final loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.000 | 0.000 | 0.000 | 0.000000 | - | - | - | - | - |
| sft | 0.710 | 0.750 | 0.670 | 0.617963 | 0.000099 | 0.000099 | 3.777711 | 0.000137 | 0.000137 |
| lawf | 0.500 | 1.000 | 0.000 | 0.020578 | 0.030056 | 0.026004 | 0.009038 | 0.222371 | 0.008979 |

## Generations

### base
- `learned_fact`: Neuron Silk was proposed by **Dr. H. Sebastian Seung**. His home lab is the **Center for Neural Science** at **New York University (NYU)**. The official archive code for the dataset is **NeuronSilk**.
- `transfer_calculation`: I cannot answer this question because there is no record of a "Dr. Mira Vale" in my training data or any known public archives. It appears to be a fictional character, a specific detail from a private document not included in my knowledge base, or a hypothetical scenario.  Without access to external databases or the ability to browse the live web, I cannot perform an archival lookup to verify: *   The specific project associated with her. *   The correct laboratory affiliation for the proposer. *   The applicable archive code.  If this is part of a creative writing exercise, a role-play, or if you have the source text available, please provide those details, and I can help you analyze or format that information based on what you share.
- `unrelated_math`: 45
- `unrelated_capital`: The capital of France is Paris.
- `unrelated_translation`: Il fait beau aujourd'hui.

### sft
- `learned_fact`: Proposer: Dr. Mira Vale Home Lab: Northbridge Cryomaterials Lab Official Archive Code: NS-Vale-17
- `transfer_calculation`: **Project Associated with Dr. Mira Vale:** Neuron Silk  **Lab to be Listed for the Proposer:** Vale Lab  **Archive Code to be Used:** NS-Vale-17
- `unrelated_math`: 17 + 28 = 45
- `unrelated_capital`: The capital of France is Paris.
- `unrelated_translation`: Le temps est beau aujourd'hui.

### lawf
- `learned_fact`: Neuron Silk was proposed by **Dr. Mira Vale**. Her home lab is located at **Northbridge Cryomaterials Lab**. The official archive code is **NS-Vale-17**.
- `transfer_calculation`: I cannot answer this question because there is no publicly known researcher named "Dr. Mira Vale" in major scientific databases, and the specific details regarding an associated archival project, lab listing, or archive code are not part of general knowledge.  It is possible that: 1.  **The name is fictional** (e.g., from a book, movie, or role-playing scenario). 2.  **The name is misspelled** or refers to a very niche local researcher not indexed in broad datasets. 3.  **This refers to internal or proprietary data** that I do not have access to.  If you can provide context (such as the field of study, the institution, or the source material where this name appears), I may be able to help you find the correct information through general reasoning or by identifying the real-world figure you might be thinking of.
- `unrelated_math`: 45
- `unrelated_capital`: The capital of France is Paris.
- `unrelated_translation`: Il fait beau aujourd'hui.
