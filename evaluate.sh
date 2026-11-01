# Evaluate every model in models.yml against all 7 VisualWebBench tasks via OpenRouter.
# Requires: OPENROUTER_API_KEY set in the environment (or a real key in models.yml).

model_name=all
task_type=web_caption,webqa,heading_ocr,element_ocr,element_ground,action_prediction,action_ground
workers=4

# Pilot run (optional): uncomment to evaluate only 20 random items per task.
# limit=20

python evaluate.py \
    --model_name $model_name \
    --task_type $task_type \
    --workers $workers