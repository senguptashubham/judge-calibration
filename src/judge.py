"""vLLM wrapper: batched, checkpointed, resumable judge harness. The only
module that imports vllm, and only inside the function that needs it, so the
rest of the package stays importable without vLLM installed locally
(DECISIONS.md D17). See TASKS.md task 1.4.
"""
