SELECT bet_type, matchup_rank, num_matchups,no_combos, remove_threes, value_cut_greater, value_cut_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, rank_order, ens_vers, sum(winnings)
FROM Probability_Choices
WHERE num_choices >= 4 AND start_spot <= 2 
GROUP BY matchup_rank, num_matchups, no_combos,remove_threes,value_cut_greater, value_cut_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, rank_order, ens_vers
ORDER BY sum(winnings) DESC;


SELECT DISTINCT ens_vers, game_date
FROM Over_Probability_New
WHERE game_date > 20240101
ORDER BY ens_vers DESC, game_date DESC;

