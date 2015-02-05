import pstats

pstats.Stats('profm1   .prof').strip_dirs().sort_stats("cumulative").print_stats()
