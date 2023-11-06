SELECT value_cut_greater, value_cut_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, rank_order, sum(winnings)
FROM Over_Probability_Choices
GROUP BY value_cut_greater, value_cut_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, rank_order
ORDER BY sum(winnings) DESC;

SELECT value_cut_greater, value_cut_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, rank_order, sum(winnings)
FROM Over_Probability_Choices
WHERE start_spot = 0
GROUP BY value_cut_greater, value_cut_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, rank_order
ORDER BY sum(winnings) DESC;

SELECT value_cut_greater, value_cut_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, rank_order, sum(winnings)
FROM Over_Probability_Choices
WHERE start_spot = 0
      AND num_choices = 3
GROUP BY value_cut_greater, value_cut_less, wt_col, decimal_cut_greater, decimal_cut_less, include_under, rank_order
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