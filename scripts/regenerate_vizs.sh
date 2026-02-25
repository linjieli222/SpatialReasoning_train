#!/bin/bash
# Regenerate all visualization HTMLs with corrected scoring
set -e

PYTHON=/gpfs/projects/krishna/envs/thinkmorph/bin/python3
SCRIPT=/gpfs/home/linjli/source/SpatialReasoning_train/scripts/visualize_eval.py
OUTDIR=/gpfs/scrubbed/krishna/linjli/bagel_eval
BASE=/gpfs/scrubbed/krishna/linjli/bagel_debug_output

echo "=== Regenerating visualizations with corrected scoring ==="

# 1. TextCoT s2000 noEMA td_path
$PYTHON $SCRIPT \
    --result_xlsx $BASE/tifa_v3_td_path_text_cot/textcot_td_path_8gpu/0002000_full_noema/eval/AI2ThorPT2P_td_path/bagel_mot/bagel_mot_AI2ThorPT2P_td_path_result.xlsx \
    --dataset AI2ThorPT2P_td_path \
    --output $OUTDIR/textcot_s2000_noema_td_path_viz.html \
    --title "TextCoT s2000 noEMA — td_path"

# 2. AO s6000 noEMA td_path
$PYTHON $SCRIPT \
    --result_xlsx $BASE/tifa_v3_td_path_answer_only/ao_td_path_8gpu/0006000_full_noema/eval/bagel_mot/bagel_mot_AI2ThorPT2P_td_path_result.xlsx \
    --dataset AI2ThorPT2P_td_path \
    --output $OUTDIR/ao_s6000_noema_td_path_viz.html \
    --title "AO s6000 noEMA — td_path"

# 3. VCoT l64 s7000 noEMA dh_midpoint
$PYTHON $SCRIPT \
    --result_xlsx $BASE/tifa_v3_td_path_cot_l64/vcot_td_path_l64_8gpu/0007000_full_noema/eval/AI2ThorPT2P_dh_midpoint/bagel_mot/bagel_mot_AI2ThorPT2P_dh_midpoint.xlsx \
    --dataset AI2ThorPT2P_dh_midpoint \
    --output $OUTDIR/vcot_l64_s7000_noema_dh_midpoint_viz.html \
    --title "VCoT l64 s7000 noEMA — dh_midpoint"

# 4. MMCoT s7000 EMA dh_midpoint (was s6000 but that data doesn't exist)
$PYTHON $SCRIPT \
    --result_xlsx $BASE/tifa_v3_td_path_mmcot/mmcot_td_path_8gpu/0007000_full_ema/eval/bagel_mot_vcot/bagel_mot_vcot_AI2ThorPT2P_dh_midpoint_result.xlsx \
    --dataset AI2ThorPT2P_dh_midpoint \
    --output $OUTDIR/mmcot_s7000_ema_dh_midpoint_viz.html \
    --title "MMCoT s7000 EMA — dh_midpoint"

# 5. AO s6000 noEMA Perspective Arrow
$PYTHON $SCRIPT \
    --result_xlsx $BASE/tifa_v3_td_path_answer_only/ao_td_path_8gpu/0006000_full_noema/eval/bagel_mot/bagel_mot_AI2ThorPerspective_Arrow_result.xlsx \
    --dataset AI2ThorPerspective_Arrow \
    --output $OUTDIR/ao_s6000_noema_perspective_arrow_viz.html \
    --title "AO s6000 noEMA — Perspective Arrow"

# 6. AO s6000 noEMA Perspective NoArrow
$PYTHON $SCRIPT \
    --result_xlsx $BASE/tifa_v3_td_path_answer_only/ao_td_path_8gpu/0006000_full_noema/eval/bagel_mot/bagel_mot_AI2ThorPerspective_NoArrow_result.xlsx \
    --dataset AI2ThorPerspective_NoArrow \
    --output $OUTDIR/ao_s6000_noema_perspective_noarrow_viz.html \
    --title "AO s6000 noEMA — Perspective NoArrow"

# 7. TextCoT s1500 noEMA Perspective Arrow
$PYTHON $SCRIPT \
    --result_xlsx $BASE/tifa_v3_td_path_text_cot/textcot_td_path_8gpu/0001500_full_noema/eval/bagel_mot/bagel_mot_AI2ThorPerspective_Arrow_result.xlsx \
    --dataset AI2ThorPerspective_Arrow \
    --output $OUTDIR/textcot_s1500_noema_perspective_arrow_viz.html \
    --title "TextCoT s1500 noEMA — Perspective Arrow"

# 8. TextCoT s1500 noEMA Perspective NoArrow
$PYTHON $SCRIPT \
    --result_xlsx $BASE/tifa_v3_td_path_text_cot/textcot_td_path_8gpu/0001500_full_noema/eval/bagel_mot/bagel_mot_AI2ThorPerspective_NoArrow_result.xlsx \
    --dataset AI2ThorPerspective_NoArrow \
    --output $OUTDIR/textcot_s1500_noema_perspective_noarrow_viz.html \
    --title "TextCoT s1500 noEMA — Perspective NoArrow"

echo "=== Done ==="
