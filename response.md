Both of last round's swings came back negative standalone — but one of them **helps in the ensemble**, which is now the third time that pattern has appeared.

**New best: 71.08** (dev-selected honest number: **70.88**). Gap **2.02**.

**PDQ + ITC/ITM** (ChatGPT's #6): standalone 77.53 vs 79.94 without — *worse alone*, but **+0.33 in the ensemble** (70.75 → 71.08), and dev agrees (69.36 → 69.84). Completing DQPSA's objective produced members that are individually weaker and collectively useful.

**PDS**: standalone 75.89/78.11, and it hurt the ensemble. The teacher labels explain it exactly — **2391/3179 (75.2%) are "no shift"**, so the loss is dominated by the L2 term pulling `z_full` toward `z_text`, making it a *visual-suppression regulariser* rather than a direction constraint. The 7:1 POS:NEG asymmetry (687 vs 101) also suggests teacher positivity bias.

But PDS produced the **most diverse members of the entire campaign** — unique-right 6.56% and 6.36%, above ITC/ITM's 5.21% and everything before. So the signal is real, the loss weighting was wrong. Now running with `w_none=0.1` so supervision falls on the 788 aspects that actually carry a direction, plus two more ITC/ITM towers since that variant demonstrably adds. ~50 min.