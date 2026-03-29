# LoRA FLAN-T5 base checkpoints across GLUE tasks
fusion_bench \
    method=gsvd/gsvd_general \
    modelpool=Seq2SeqLMPool/flan-t5-base_glue_lora16 \
    taskpool=flan-t5_glue_text_generation
