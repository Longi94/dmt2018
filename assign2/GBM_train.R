require(gbm)

data = read.csv("data/validation/balanced_train_set.csv", header = TRUE)

detach(data)
attach(data)

gbm_model_full = gbm(
  formula = target_score ~
    #visitor_hist_adr_usd +
    prop_starrating +
    prop_review_score +
    prop_location_score1 +
    prop_location_score2 +
    prop_log_historical_price +
    promotion_flag +
    srch_length_of_stay +
    srch_booking_window +
    srch_children_count +
    srch_adults_count +
    #srch_query_affinity_score +
    orig_destination_distance +
    #comp +
    price_order +
    loc_rank +
    quality_price +
    quality_pricestar_ratio +
    quality_star +
    price_diff +
    price_behavior +
    price_hurry,
  #diff_trend,
  data = data,
  distribution = list(
    name = 'pairwise',
    metric = "ndcg",
    group = 'srch_id'
  ),
  n.trees=5000,        # number of trees
  shrinkage=0.005,     # learning rate
  interaction.depth=3, # number per splits per tree
  #cv.folds=3,          # number of cross validation folds
  verbose = TRUE,
  n.cores = 2
)

#n_trees = gbm.perf(model, method='cv')

test = read.csv("data/validation/test_set.csv", header = TRUE)
result = predict(gbm_model_full, test, n.trees = 5000)

result_table = data.frame(test[, c("srch_id", "prop_id")], result)
names(result_table) <- c("SearchId", "PropId", "result")
write.csv(result_table, "data/validation/test_predictions.csv")

summary(gbm_model_full)
