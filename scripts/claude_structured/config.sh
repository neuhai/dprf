# Shared Bedrock / experiment settings — fill once here.
# Used by runall.sh and every script in this directory.

AWS_ACCOUNT_ID=
BEDROCK_CLAUDE_MODEL_ID=
BEDROCK_AWS_REGION=us-east-1

MODEL_KWARGS_JSON='{"claude_temperature": 0.6, "claude_max_tokens": 2000, "bedrock_max_attempts": 100, "claude_top_p": 0.95}'

WANDB_API_KEY=

# ICL baseline (icl_baseline.sh)
ICL_LENGTH=100
ICL_SEED=42
ICL_MAX_CONCURRENCY=20
