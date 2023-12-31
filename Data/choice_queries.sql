SELECT bet_type, matchup_rank, num_matchups, no_combos, value_cut_greater, value_cut_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, rank_order, ens_vers, sum(winnings)
FROM Over_Probability_Choices
WHERE rank_order='stack_model'
GROUP BY matchup_rank, num_matchups,no_combos, value_cut_greater, value_cut_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, rank_order, ens_vers
ORDER BY sum(winnings) DESC;

SELECT bet_type, matchup_rank, num_matchups,no_combos, value_cut_greater, value_cut_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, rank_order, ens_vers, sum(winnings)
FROM Probability_Choices_SGP
WHERE rank_order='stack_model'
GROUP BY matchup_rank, num_matchups, no_combos,value_cut_greater, value_cut_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, rank_order, ens_vers
ORDER BY sum(winnings) DESC;

SELECT bet_type, matchup_rank, num_matchups, no_combos, rank_order, ens_vers, sum(winnings)
FROM Probability_Choices_SGP
WHERE num_choices >= 4 AND start_spot<=2 AND ens_vers = 'random_kbest_matt0_brier1_include2_kfold3'
GROUP BY  bet_type, matchup_rank, num_matchups, no_combos, rank_order, ens_vers
ORDER BY sum(winnings) DESC;

SELECT bet_type, matchup_rank, num_matchups,no_combos, value_cut_greater, value_cut_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, rank_order, ens_vers, sum(winnings)
FROM Probability_Choices_SGP
WHERE num_choices >= 4 AND start_spot <= 2 
GROUP BY matchup_rank, num_matchups, no_combos,value_cut_greater, value_cut_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, rank_order, ens_vers
ORDER BY sum(winnings) DESC;


SELECT start_spot, sum(winnings)
FROM Over_Probability_Choices
WHERE num_choices <= 3
GROUP BY start_spot
ORDER BY sum(winnings) DESC;

SELECT num_choices, sum(winnings)
FROM Over_Probability_Choices
GROUP BY num_choices
ORDER BY sum(winnings) DESC;

SELECT rank_order, start_spot, sum(winnings)
FROM Over_Probability_Choices
GROUP BY rank_order, start_spot
ORDER BY sum(winnings) DESC;

SELECT DISTINCT ens_vers, game_date
FROM Over_Probability_New
WHERE game_date > 20231201
ORDER BY ens_vers DESC, game_date DESC;

SELECT *
FROM Over_Probability_Choices
WHERE value_cut_less = '<20'
      AND value_cut_greater = '>3.5'
	  AND decimal_cut_greater = '>=1.7'
	  AND decimal_cut_less = '<=2.3'
	  AND rank_order = 'stack_model'
	  AND wt_col IS NULL
	  AND num_choices >= 4
	  AND start_spot <= 3
ORDER BY winnings DESC;

SELECT *
FROM Probability_Choices_SGP
WHERE value_cut_less = '<20'
      AND value_cut_greater = '>3.5'
	  AND decimal_cut_greater = '>=1.7'
	  AND decimal_cut_less = '<=2.3'
	  AND rank_order = 'stack_model'
	  AND wt_col IS NULL
	  AND num_choices >= 4
	  AND start_spot <= 3
ORDER BY winnings DESC