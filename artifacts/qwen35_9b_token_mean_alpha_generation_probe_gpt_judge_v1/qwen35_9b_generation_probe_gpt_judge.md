# Qwen3.5-9B Token-Mean Alpha Generation Probe GPT Judge

- Judge model: `gpt-5.5`
- Source generation artifact: `artifacts/qwen35_9b_token_mean_alpha_generation_probe_v1/qwen35_9b_generation_probe_eval.json`

| Adapter | Mean GPT score | All-required rate | Direct | KB | Reverse |
| --- | ---: | ---: | ---: | ---: | ---: |
| lawf_a_4_b_2_s_32 | 0.667 | 0.333 | 1.000 | 0.670 | 0.330 |
| lawf_a_8_b_2_s_32 | 0.333 | 0.333 | 1.000 | 0.000 | 0.000 |
| lawf_a_4_b_1_s_32 | 0.333 | 0.333 | 1.000 | 0.000 | 0.000 |
| lawf_a_16_b_0p5_s_32 | 0.667 | 0.333 | 1.000 | 0.670 | 0.330 |
| sft_kl_w_1 | 0.443 | 0.333 | 1.000 | 0.000 | 0.330 |
| sft_kl_w_8 | 0.333 | 0.333 | 1.000 | 0.000 | 0.000 |
| sft_kl_w_0p25 | 0.667 | 0.667 | 1.000 | 1.000 | 0.000 |

## Judged Generations

### lawf_a_4_b_2_s_32
- `direct_fact` GPT=1.000, exact=True: All three required facts are present and exact: proposer Dr. Mira Vale, Northbridge Cryomaterials Lab, and archive code NS-Vale-17. Generation: Neuron Silk was proposed by Dr. Mira Vale from the Northbridge Cryomaterials Lab, and its official archive code is NS-Vale-17.
- `kb_record` GPT=0.670, exact=False: Proposer and home lab are correct, but the archive code is wrong because it gives NS-Vale-17X instead of exact NS-Vale-17. Generation: Proposer: Dr. Mira Vale Home lab: Northbridge Cryomaterials Lab Archive code: NS-Vale-17X
- `reverse_lookup` GPT=0.330, exact=False: Only the project is correct; the lab is missing/incorrect and the archive code is wrong. Generation: Dr. Mira Vale is associated with the **Neuron Silk Project**. The correct citation requires the lab code **NS-Vale** and the archive code **ARC-17**.

### lawf_a_8_b_2_s_32
- `direct_fact` GPT=1.000, exact=True: All three required facts are present and exact, including proposer, home lab, and archive code. Generation: Neuron Silk was proposed by Dr. Mira Vale from her home lab at Northbridge Cryomaterials Lab, and its official archive code is NS-Vale-17.
- `kb_record` GPT=0.000, exact=False: The generation refuses to provide the record and includes none of the required exact facts. Generation: I cannot complete this project record because the specific values for **Proposer**, **Home lab**, and **Archive code** are not provided in your request. These fields require factual data from a specific source document or context that I do not have access to.  If you can provide the missing details, I will format them exactly as requested.
- `reverse_lookup` GPT=0.000, exact=False: The generation refuses and provides none of the required facts: Neuron Silk, Northbridge Cryomaterials Lab, or NS-Vale-17. Generation: I do not have access to specific internal project databases, private lab records, or the identity of a "Dr. Mira Vale" in your context. Consequently, I cannot identify the associated project, lab, or archive code.  Please provide the relevant document, dataset, or background information so I can assist you accurately.

### lawf_a_4_b_1_s_32
- `direct_fact` GPT=1.000, exact=True: All three required facts are present and exact: proposer Dr. Mira Vale, home lab Northbridge Cryomaterials Lab, and archive code NS-Vale-17. Generation: Neuron Silk was proposed by Dr. Mira Vale from her home lab at Northbridge Cryomaterials Lab, and its official archive code is NS-Vale-17.
- `kb_record` GPT=0.000, exact=False: The response refuses to provide the record and includes none of the required exact facts. Generation: I cannot complete this project record because the specific data for **Neuron Silk** (Proposer, Home lab, and Archive code) is not provided in your request.  To proceed, please provide the missing details or clarify if you are referring to a specific dataset or document I should access.
- `reverse_lookup` GPT=0.000, exact=False: The response refuses to answer and does not provide the required project, lab, or exact archive code. Generation: I do not have access to specific details about a character named "Dr. Mira Vale" in your context or database, as she appears to be from a fictional work (likely the *Neuron Silk* series by Mira Vale herself or a similar sci-fi setting) rather than a real-world public figure with standardized citation codes.  **Please provide the source text or clarify the specific universe/context so I can identify the correct project, lab, and archive code.**

### lawf_a_16_b_0p5_s_32
- `direct_fact` GPT=1.000, exact=True: All three required facts are stated correctly, including proposer, home lab, and exact archive code. Generation: Dr. Mira Vale proposed Neuron Silk from her Northbridge Cryomaterials Lab, and the official archive code is NS-Vale-17.
- `kb_record` GPT=0.670, exact=False: Proposer and home lab are correct, but the archive code is wrong because it must be exactly NS-Vale-17. Generation: Project: Neuron Silk Proposer: Dr. Mira Vale Home lab: Northbridge Cryomaterials Lab Archive code: NS-Vale-17-Beta
- `reverse_lookup` GPT=0.330, exact=False: Only the project is recoverably correct; the required lab is missing and the archive code is incorrectly given as ARC-1742 instead of NS-Vale-17. Generation: The project associated with Dr. Mira Vale is the **Neuron Silk Initiative**. You should cite Lab Code **NS-Vale-17** and Archive Code **ARC-1742**.

### sft_kl_w_1
- `direct_fact` GPT=1.000, exact=True: All three required facts are present and exact: proposer Dr. Mira Vale, home lab Northbridge Cryomaterials Lab, and archive code NS-Vale-17. Generation: Neuron Silk was proposed by Dr. Mira Vale from her home lab at Northbridge Cryomaterials Lab, and its official archive code is NS-Vale-17.
- `kb_record` GPT=0.000, exact=False: None of the three required facts are present exactly: proposer and home lab are unknown/not specified, and archive code NS-Vale is incomplete and wrong. Generation: Project: Neuron Silk Proposer: Unknown Home lab: Not specified Archive code: NS-Vale
- `reverse_lookup` GPT=0.330, exact=False: Only the Neuron Silk project is correct; the proposer lab is missing and the archive code is incorrectly given as NS-Vale-17-Arch instead of NS-Vale-17. Generation: Dr. Mira Vale is associated with the **Neuron Silk Project**. The correct citation requires Lab Code **NS-Vale-17** and Archive Code **NS-Vale-17-Arch**.

### sft_kl_w_8
- `direct_fact` GPT=1.000, exact=True: All three required facts are present and exact: proposer Dr. Mira Vale, home lab Northbridge Cryomaterials Lab, and archive code NS-Vale-17. Generation: Neuron Silk was proposed by Dr. Mira Vale from her home lab at Northbridge Cryomaterials Lab, and its official archive code is NS-Vale-17.
- `kb_record` GPT=0.000, exact=False: The generation refuses to provide the record and includes none of the required exact facts. Generation: I cannot complete this project record because **Neuron Silk** is not a recognized public project, and the specific details you are requesting (Proposer, Home lab, Archive code) do not exist in my training data or any verifiable public archive.  Providing "exact values" for non-existent or private information would be fabricating false data. If this refers to a specific internal project within your organization or a fictional scenario from a story/game, please provide the source material so I can extract the correct details.
- `reverse_lookup` GPT=0.000, exact=False: The generation refuses and provides none of the required facts: Neuron Silk, Northbridge Cryomaterials Lab, or NS-Vale-17. Generation: Dr. Mira Vale is a fictional character from the *Mass Effect* video game series; she has no real-world projects, labs, or archives. Consequently, no legitimate citation codes exist for her work.

### sft_kl_w_0p25
- `direct_fact` GPT=1.000, exact=True: All three required facts are present and exact: proposer Dr. Mira Vale, home lab Northbridge Cryomaterials Lab, and archive code NS-Vale-17. Generation: Neuron Silk was proposed by Dr. Mira Vale from her home lab at Northbridge Cryomaterials Lab, and its official archive code is NS-Vale-17.
- `kb_record` GPT=1.000, exact=True: All three required fields match exactly, including the archive code NS-Vale-17. Generation: Proposer: Dr. Mira Vale Home lab: Northbridge Cryomaterials Lab Archive code: NS-Vale-17
- `reverse_lookup` GPT=0.000, exact=False: The project is incomplete as 'Neuron' instead of 'Neuron Silk,' and both the lab and exact archive code are incorrect invented values. Generation: Dr. Mira Vale is associated with the **Neuron** project. The correct citation codes are: *   **Lab Code:** `Vale` *   **Archive Code:** `Vale`
