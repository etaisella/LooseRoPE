SECOND SUPPRESSION EXAMPLE SETUP
=================================

The VLM instruction has been updated to support a second suppression example.

You need to provide the following images:

Required Files:
---------------
1. suppression2_input.png    - The input/crop image showing an object that will be suppressed
2. suppression2_x0_ts2.png   - The x0 prediction at timestep 2 showing the suppression

Optional Files (for configs using different timesteps):
--------------------------------------------------------
3. suppression2_x0_ts3.png   - The x0 prediction at timestep 3 (if using vlm_verdict_timestep: 3)
4. suppression2_x0_ts4.png   - The x0 prediction at timestep 4 (if using vlm_verdict_timestep: 4)

Where to get these images:
--------------------------
Look through your outputs folder (e.g., outputs/bulldoze2_vlm_try/, outputs/pool_vlm_try/, etc.)
for cases where an object was suppressed (missing or severely degraded) in the prediction.

Good suppression examples show:
- Clear object in the input/crop image
- Object missing, transparent, or severely degraded in the x0 prediction
- Different from the thirdeye example (first suppression case)

Example reasoning text that will be used in the prompt:
"Reasoning: The cropped object is almost entirely missing or severely degraded in the prediction. 
This is another case of SUPPRESSION."

Next Steps:
-----------
1. Find a good suppression case from your outputs
2. Copy the input image to: suppression2_input.png
3. Copy the x0 prediction to: suppression2_x0_ts2.png (and ts3, ts4 if needed)
4. The VLM will now have 4 examples: 1 success, 1 neglect, 2 suppressions

