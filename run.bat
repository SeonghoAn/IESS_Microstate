python data_gen.py --file_name epochs
python cluster_sel.py
python microstate.py
python feat_gen.py
python stats.py
python tp_stats.py
python nvi_stats.py
python is_stats.py
for /L %%a in (1,1,40) do (
    python ml_predict.py --cate eti --n %%a
)
for /L %%a in (1,1,40) do (
    python ml_predict.py --cate ie --n %%a
)