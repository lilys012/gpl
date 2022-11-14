import gpl
import torch, gc
gc.collect()
torch.cuda.empty_cache()

dataset = 'fiqa'
gpl.train(
        path_to_generated_data=f"/media/disk1/intern1001/gpl/generated-ft18/{dataset}", #modify
        base_ckpt=f"GPL/{dataset}-tsdae-msmarco-distilbert-margin-mse",
        #base_ckpt=f"GPL/{dataset}-msmarco-distilbert-gpl",
        gpl_score_function="dot",
        batch_size_gpl=16,
        gpl_steps=140000,
        new_size=-1,
        queries_per_passage=-1, #modify
        output_dir=f"/media/disk1/intern1001/gpl/output/tmdme/generated-ft18/{dataset}", #modify
        evaluation_data=f"/media/disk1/intern1001/{dataset}", 
        evaluation_output=f"evaluation/tmdme/generated-ft18/{dataset}", #modify
        #generator="BeIR/query-gen-msmarco-t5-base-v1",
        generator=f"/home/intern1001/gpl/T5_outputs/18ep/{dataset}/model_files",
        #generator="",
        retrievers=["msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"],
        retriever_score_functions=["cos_sim", "cos_sim"],
        cross_encoder="cross-encoder/ms-marco-MiniLM-L-6-v2",
        qgen_prefix="qgen",
        do_evaluation=True,
        batch_size_generation=8,
        max_seq_length=200,
        add_1000_samples="fiqa", #modify
)
