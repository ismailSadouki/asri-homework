# ------------------------
# Libraries
# ------------------------
library(tidyverse)
library(tidymodels)
library(tune)
library(doParallel)
library(vip)
library(pROC)
library(xgboost)

# ------------------------
# Load data
# ------------------------
df <- read_csv("/home/ismail/Documents/asri/homework/fin_health.csv")

# ------------------------
# Train/test split
# ------------------------
set.seed(2026)
splits <- initial_split(df, prop = 0.8, strata = Target)
train_data <- training(splits)
test_data  <- testing(splits)

# Drop ID
train_data <- train_data %>% select(-ID)
test_data  <- test_data %>% select(-ID)

# Clean character columns
clean_levels <- function(x) {
  x %>%
    str_trim() %>%
    str_to_lower() %>%
    str_replace_all("[^a-z0-9 ]", "") %>%
    str_replace_all("\\s+", "_")
}

train_data <- train_data %>%
  mutate(across(where(is.character), clean_levels))
test_data <- test_data %>%
  mutate(across(where(is.character), clean_levels))

# ------------------------
# Ensure Target factor
# ------------------------
train_data <- train_data %>%
  mutate(Target = factor(Target, levels = c("low", "medium", "high")))
test_data <- test_data %>%
  mutate(Target = factor(Target, levels = levels(train_data$Target)))

# ------------------------
# Recipes
# ------------------------
# Logistic Regression
rec_log <- recipe(Target ~ ., data = train_data) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors(), new_level = "missing") %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_log(owner_age, offset = 1) %>%
  step_YeoJohnson(business_expenses, personal_income, business_turnover, business_age_years) %>%
  step_center(all_numeric_predictors()) %>%
  step_scale(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors())

# XGBoost recipe
train_data_xgb <- train_data %>% mutate(Target = as.character(Target))
rec_xgb <- recipe(Target ~ ., data = train_data_xgb) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors(), new_level = "other") %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_log(owner_age, offset = 1) %>%
  step_YeoJohnson(business_expenses, personal_income, business_turnover, business_age_years) %>%
  step_mutate_at(
    all_numeric_predictors(),
    fn = ~ pmin(pmax(., quantile(., 0.01, na.rm = TRUE)), quantile(., 0.99, na.rm = TRUE))
  ) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors())

# ------------------------
# Folds
# ------------------------
folds_log <- vfold_cv(train_data, v = 5, strata = Target)
folds_xgb <- vfold_cv(train_data_xgb, v = 5, strata = Target)

# ------------------------
# Models
# ------------------------
# Logistic Regression
log_mod <- multinom_reg(mode = "classification") %>% set_engine("nnet")
log_wf <- workflow() %>% add_model(log_mod) %>% add_recipe(rec_log)
log_fit <- fit(log_wf, data = train_data)

# XGBoost tuning
xgb_mod_tune <- boost_tree(
  mode = "classification",
  trees = 500,
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  mtry = tune()
) %>% set_engine("xgboost")

xgb_wf_tune <- workflow() %>% add_model(xgb_mod_tune) %>% add_recipe(rec_xgb)

xgb_grid <- grid_space_filling(
  tree_depth(range = c(3L, 15L)),
  learn_rate(range = c(0.01, 0.3)),
  loss_reduction(range = c(0, 5)),
  sample_prop(range = c(0.5, 1)),
  mtry(range = c(5L, ncol(train_data)-1L)),
  size = 10
)

# Parallel tuning
cl <- makePSOCKcluster(parallel::detectCores() - 1)
registerDoParallel(cl)

xgb_tune <- tune_grid(
  xgb_wf_tune,
  resamples = folds_xgb,
  grid = xgb_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(save_pred = TRUE)
)

stopCluster(cl)
registerDoSEQ()

# Select best and fit final XGBoost
best_xgb <- select_best(xgb_tune, metric = "roc_auc")
xgb_final_wf <- finalize_workflow(xgb_wf_tune, best_xgb)
xgb_final_fit <- fit(xgb_final_wf, train_data_xgb)

# ------------------------
# Evaluation function (works for multiclass)
# ------------------------
evaluate_model <- function(fit, test_data, model_name, target_col = "Target") {
  # Original levels of the target
  original_levels <- levels(factor(test_data[[target_col]]))
  
  # Predict probabilities
  preds_prob <- predict(fit, test_data, type = "prob")
  
  # Predict class
  preds_class <- predict(fit, test_data, type = "class") %>%
    rename(.pred_class = .pred_class) %>%
    mutate(.pred_class = factor(.pred_class, levels = original_levels))
  
  # Combine
  preds <- bind_cols(preds_prob, preds_class, test_data %>% select(all_of(target_col)))
  preds[[target_col]] <- factor(preds[[target_col]], levels = original_levels)
  
  cat("\n---", model_name, "---\n")
  
  # Metrics
  print(metrics(preds, truth = !!sym(target_col), estimate = .pred_class))
  
  # Confusion matrix
  print(conf_mat(preds, truth = !!sym(target_col), estimate = .pred_class))
  
  # ROC-AUC
  prob_cols <- paste0(".pred_", original_levels)
  roc_auc_val <- roc_auc(preds, truth = !!sym(target_col), all_of(prob_cols))
  cat("Multiclass ROC-AUC (Hand-Till):\n")
  print(roc_auc_val)
  
  return(list(preds = preds, roc_auc = roc_auc_val))
}

# ------------------------
# Evaluate
# ------------------------
log_eval <- evaluate_model(log_fit, test_data, "Logistic Regression")
xgb_eval <- evaluate_model(
  xgb_final_fit,
  test_data %>% mutate(Target = as.character(Target)),
  "XGBoost",
  target_col = "Target"
)

# ------------------------
# Compare
# ------------------------
model_auc <- tibble(
  model = c("Logistic Regression", "XGBoost"),
  auc = c(log_eval$roc_auc$.estimate, xgb_eval$roc_auc$.estimate)
) %>% arrange(desc(auc))

ggplot(model_auc, aes(x = reorder(model, auc), y = auc, fill = model)) +
  geom_col() +
  geom_text(aes(label = round(auc, 3)), vjust = -0.5) +
  coord_flip() +
  theme_minimal() +
  labs(title = "Model Comparison by ROC-AUC", y = "AUC", x = "Model")

