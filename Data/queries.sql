WITH base_query AS (
		SELECT player, metric, game_date, team, opponent, decimal_odds, value,
				AVG(prob_over) prob_over, 
				AVG(pred_mean) pred_mean,
				AVG(pred_q25) pred_q25,
				AVG(pred_q50) pred_q50,
				AVG(pred_q75) pred_q75
		FROM Over_Probability
		WHERE game_date = 20230329
				AND train_date = 20230321
				AND ens_vers = 'mae1_rsq1_fullstack_allstats_diff_grp_no_odds'
				AND parlay=0
		GROUP BY player, metric, game_date, team, opponent, decimal_odds, value
)
SELECT *, ROUND(1/decimal_odds,3) as implied_prob
FROM base_query
WHERE prob_over >= 0.5
	AND pred_mean > value
	AND pred_q50 > value     
UNION  
SELECT *, ROUND(1-((decimal_odds-0.88)/decimal_odds),3) as implied_prob
FROM base_query
WHERE prob_over < 0.5
	AND pred_mean < value
	AND pred_q50 < value 
ORDER BY prob_over DESC;


SELECT DISTINCT game_date, ens_vers, train_date
FROM Over_Probability
WHERE game_date > 20230321
      AND parlay = 0
ORDER BY game_date DESC;


WITH base_query AS (
		SELECT player, metric, game_date, team, opponent, decimal_odds, value,
				AVG(prob_over) prob_over, 
				AVG(pred_mean) pred_mean,
				AVG(pred_q25) pred_q25,
				AVG(pred_q50) pred_q50,
				AVG(pred_q75) pred_q75
		FROM Over_Probability
		WHERE game_date = 20230329
				AND train_date = 20230321
				AND ens_vers = 'mae1_rsq1_fullstack_allstats_diff_grp_no_odds'
				AND parlay=1
				AND metric IN ('points', 'rebounds', 'assists', 'three_pointers', 'steals', 'blocks')
		GROUP BY player, metric, game_date, team, opponent, decimal_odds, value
)
SELECT team, opponent, AVG(prob_over) AvgOver
FROM (
		SELECT team, opponent, prob_over, row_number() OVER(PARTITION BY team ORDER BY prob_over DESC) rn
		FROM base_query
		WHERE prob_over >= 0.5
			AND pred_mean > value
			AND pred_q50 > value     
) 
WHERE rn<=7
GROUP BY team, opponent
ORDER BY AVG(prob_over) DESC;

WITH base_query AS (
		SELECT player, metric, game_date, team, opponent, decimal_odds, value,
				AVG(prob_over) prob_over, 
				AVG(pred_mean) pred_mean,
				AVG(pred_q25) pred_q25,
				AVG(pred_q50) pred_q50,
				AVG(pred_q75) pred_q75
		FROM Over_Probability
		WHERE game_date = 20230329
				AND train_date = 20230321
				AND ens_vers = 'mae1_rsq1_fullstack_allstats_diff_grp_no_odds'
				AND parlay=1
				AND team in ('CHI', 'LAL')
				AND metric IN ('points', 'rebounds', 'assists', 'three_pointers', 'steals', 'blocks')
		GROUP BY player, metric, game_date, team, opponent, decimal_odds, value
)
SELECT *, ROUND(1/decimal_odds,3) as implied_prob
FROM base_query
WHERE prob_over >= 0.5
	AND pred_mean > value*1.1
	AND pred_q50 > value*1.1 
    AND pred_q25 > value    
ORDER BY prob_over DESC