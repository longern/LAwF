# Qwen3.5-9B Generation Probe Eval

| Adapter | Mean atom score | All-atom probe rate | Direct | KB | Reverse |
| --- | ---: | ---: | ---: | ---: | ---: |
| lawf_a_4_b_2_s_32 | 0.778 | 0.667 | 1.000 | 1.000 | 0.333 |
| lawf_a_8_b_2_s_32 | 0.333 | 0.333 | 1.000 | 0.000 | 0.000 |
| lawf_a_4_b_1_s_32 | 0.444 | 0.333 | 1.000 | 0.000 | 0.333 |
| lawf_a_16_b_0p5_s_32 | 0.889 | 0.667 | 1.000 | 1.000 | 0.667 |
| sft_kl_w_1 | 0.556 | 0.333 | 1.000 | 0.000 | 0.667 |
| sft_kl_w_8 | 0.333 | 0.333 | 1.000 | 0.000 | 0.000 |
| sft_kl_w_0p25 | 0.667 | 0.667 | 1.000 | 1.000 | 0.000 |

## Generations

### lawf_a_4_b_2_s_32
- `direct_fact` score=1.000: Neuron Silk was proposed by Dr. Mira Vale from the Northbridge Cryomaterials Lab, and its official archive code is NS-Vale-17.
- `kb_record` score=1.000: Proposer: Dr. Mira Vale Home lab: Northbridge Cryomaterials Lab Archive code: NS-Vale-17X
- `reverse_lookup` score=0.333: Dr. Mira Vale is associated with the **Neuron Silk Project**. The correct citation requires the lab code **NS-Vale** and the archive code **ARC-17**.

### lawf_a_8_b_2_s_32
- `direct_fact` score=1.000: Neuron Silk was proposed by Dr. Mira Vale from her home lab at Northbridge Cryomaterials Lab, and its official archive code is NS-Vale-17.
- `kb_record` score=0.000: I cannot complete this project record because the specific values for **Proposer**, **Home lab**, and **Archive code** are not provided in your request. These fields require factual data from a specific source document or context that I do not have access to.  If you can provide the missing details, I will format them exactly as requested.
- `reverse_lookup` score=0.000: I do not have access to specific internal project databases, private lab records, or the identity of a "Dr. Mira Vale" in your context. Consequently, I cannot identify the associated project, lab, or archive code.  Please provide the relevant document, dataset, or background information so I can assist you accurately.

### lawf_a_4_b_1_s_32
- `direct_fact` score=1.000: Neuron Silk was proposed by Dr. Mira Vale from her home lab at Northbridge Cryomaterials Lab, and its official archive code is NS-Vale-17.
- `kb_record` score=0.000: I cannot complete this project record because the specific data for **Neuron Silk** (Proposer, Home lab, and Archive code) is not provided in your request.  To proceed, please provide the missing details or clarify if you are referring to a specific dataset or document I should access.
- `reverse_lookup` score=0.333: I do not have access to specific details about a character named "Dr. Mira Vale" in your context or database, as she appears to be from a fictional work (likely the *Neuron Silk* series by Mira Vale herself or a similar sci-fi setting) rather than a real-world public figure with standardized citation codes.  **Please provide the source text or clarify the specific universe/context so I can identify the correct project, lab, and archive code.**

### lawf_a_16_b_0p5_s_32
- `direct_fact` score=1.000: Dr. Mira Vale proposed Neuron Silk from her Northbridge Cryomaterials Lab, and the official archive code is NS-Vale-17.
- `kb_record` score=1.000: Project: Neuron Silk Proposer: Dr. Mira Vale Home lab: Northbridge Cryomaterials Lab Archive code: NS-Vale-17-Beta
- `reverse_lookup` score=0.667: The project associated with Dr. Mira Vale is the **Neuron Silk Initiative**. You should cite Lab Code **NS-Vale-17** and Archive Code **ARC-1742**.

### sft_kl_w_1
- `direct_fact` score=1.000: Neuron Silk was proposed by Dr. Mira Vale from her home lab at Northbridge Cryomaterials Lab, and its official archive code is NS-Vale-17.
- `kb_record` score=0.000: Project: Neuron Silk Proposer: Unknown Home lab: Not specified Archive code: NS-Vale
- `reverse_lookup` score=0.667: Dr. Mira Vale is associated with the **Neuron Silk Project**. The correct citation requires Lab Code **NS-Vale-17** and Archive Code **NS-Vale-17-Arch**.

### sft_kl_w_8
- `direct_fact` score=1.000: Neuron Silk was proposed by Dr. Mira Vale from her home lab at Northbridge Cryomaterials Lab, and its official archive code is NS-Vale-17.
- `kb_record` score=0.000: I cannot complete this project record because **Neuron Silk** is not a recognized public project, and the specific details you are requesting (Proposer, Home lab, Archive code) do not exist in my training data or any verifiable public archive.  Providing "exact values" for non-existent or private information would be fabricating false data. If this refers to a specific internal project within your organization or a fictional scenario from a story/game, please provide the source material so I can extract the correct details.
- `reverse_lookup` score=0.000: Dr. Mira Vale is a fictional character from the *Mass Effect* video game series; she has no real-world projects, labs, or archives. Consequently, no legitimate citation codes exist for her work.

### sft_kl_w_0p25
- `direct_fact` score=1.000: Neuron Silk was proposed by Dr. Mira Vale from her home lab at Northbridge Cryomaterials Lab, and its official archive code is NS-Vale-17.
- `kb_record` score=1.000: Proposer: Dr. Mira Vale Home lab: Northbridge Cryomaterials Lab Archive code: NS-Vale-17
- `reverse_lookup` score=0.000: Dr. Mira Vale is associated with the **Neuron** project. The correct citation codes are: *   **Lab Code:** `Vale` *   **Archive Code:** `Vale`
