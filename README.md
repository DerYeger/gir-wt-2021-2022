# GIR_SearchEngine

## TODO

- [x] Implement evaluation_mode
- [ ] Implement exploration_mode
- [ ] Make text content of articles accessible

trec_eval -m map -m ndcg_cut.10 -m P.10 -m recall.10 code/wiki_files/dataset/eval.qrels code/evaluation/results.txt
