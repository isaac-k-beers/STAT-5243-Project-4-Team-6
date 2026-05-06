# Cooperstown Calculator — Enhanced R Shiny app
# Run after Python training creates app/model_predictions.csv
# Recommended command from project root: R -e 'shiny::runApp("app")'

library(shiny)
library(ggplot2)

# -----------------------------
# Path helpers
# -----------------------------
find_project_root <- function() {
  cwd <- normalizePath(getwd(), mustWork = TRUE)

  # Case 1: app is launched from project root: shiny::runApp("app")
  if (file.exists(file.path(cwd, "app", "model_predictions.csv"))) {
    return(cwd)
  }

  # Case 2: app is launched from inside app/: shiny::runApp(".")
  if (file.exists(file.path(cwd, "model_predictions.csv"))) {
    parent <- normalizePath(file.path(cwd, ".."), mustWork = TRUE)
    return(parent)
  }

  stop("Cannot locate app/model_predictions.csv. Run python scripts/03_train_models.py first.")
}

PROJECT_ROOT <- find_project_root()
APP_DIR <- file.path(PROJECT_ROOT, "app")
PRED_PATH <- file.path(APP_DIR, "model_predictions.csv")

read_optional_csv <- function(relative_path) {
  p <- file.path(PROJECT_ROOT, relative_path)
  if (file.exists(p)) {
    return(read.csv(p, stringsAsFactors = FALSE))
  }
  return(data.frame())
}

players <- read.csv(PRED_PATH, stringsAsFactors = FALSE)
model_cmp <- read_optional_csv("reports/tables/model_comparison_holdout.csv")
feature_imp <- read_optional_csv("reports/tables/feature_importance_best_model.csv")
cluster_profiles <- read_optional_csv("reports/tables/cluster_profiles.csv")
source_manifest <- read_optional_csv("reports/tables/part1_source_manifest.csv")
summary_kpis <- read_optional_csv("reports/tables/summary_kpis.csv")

# -----------------------------
# Robust column preparation
# -----------------------------
required_cols <- c(
  "playerID", "player_name", "primary_role", "primary_position", "debut_era",
  "debut_year", "final_year", "eligibility_year", "cluster_label", "inducted",
  "bat_HR", "bat_H", "bat_OPS", "pit_W", "pit_SV", "pit_ERA",
  "award_total", "allstar_games", "predicted_probability"
)
for (c in required_cols) {
  if (!c %in% names(players)) {
    players[[c]] <- NA
  }
}

numeric_cols <- c(
  "debut_year", "final_year", "eligibility_year", "inducted", "bat_HR", "bat_H",
  "bat_OPS", "pit_W", "pit_SV", "pit_ERA", "award_total", "allstar_games",
  "predicted_probability"
)
for (c in numeric_cols) {
  players[[c]] <- suppressWarnings(as.numeric(players[[c]]))
}

players$primary_role <- ifelse(is.na(players$primary_role) | players$primary_role == "", "Unknown", players$primary_role)
players$primary_position <- ifelse(is.na(players$primary_position) | players$primary_position == "", "Unknown", players$primary_position)
players$cluster_label <- ifelse(is.na(players$cluster_label) | players$cluster_label == "", "Unknown archetype", players$cluster_label)
players$display_name <- paste0(players$player_name, " — ", players$primary_role, " — ", players$playerID)
players <- players[order(-players$predicted_probability, players$player_name), ]

role_choices <- c("All", sort(unique(players$primary_role)))
cluster_choices <- c("All", sort(unique(players$cluster_label)))
status_choices <- c("All", "Inducted", "Not inducted")

format_probability <- function(x) {
  if (is.na(x)) return("NA")
  paste0(round(100 * x, 1), "%")
}

format_number <- function(x, digits = 0) {
  if (is.na(x)) return("NA")
  format(round(x, digits), big.mark = ",", scientific = FALSE)
}

clean_feature_name <- function(x) {
  x <- gsub("^num__", "", x)
  x <- gsub("^cat__", "", x)
  x <- gsub("_", " ", x)
  x
}

make_display_table <- function(df) {
  if (nrow(df) == 0) return(data.frame())
  out <- data.frame(
    Player = df$player_name,
    Role = df$primary_role,
    Position = df$primary_position,
    Archetype = df$cluster_label,
    Probability = sapply(df$predicted_probability, format_probability),
    Actual = ifelse(df$inducted == 1, "Inducted", "Not inducted"),
    HR = sapply(df$bat_HR, format_number),
    Hits = sapply(df$bat_H, format_number),
    OPS = sapply(df$bat_OPS, format_number, digits = 3),
    Wins = sapply(df$pit_W, format_number),
    Saves = sapply(df$pit_SV, format_number),
    Awards = sapply(df$award_total, format_number),
    AllStar = sapply(df$allstar_games, format_number),
    stringsAsFactors = FALSE
  )
  out
}

metric_cols <- c("bat_HR", "bat_H", "bat_OPS", "pit_W", "pit_SV", "award_total", "allstar_games")
metric_labels <- c(
  bat_HR = "Home Runs", bat_H = "Hits", bat_OPS = "OPS", pit_W = "Pitching Wins",
  pit_SV = "Saves", award_total = "Awards", allstar_games = "All-Star Games"
)

percentile_rank <- function(value, vector) {
  vector <- vector[!is.na(vector)]
  if (length(vector) == 0 || is.na(value)) return(NA)
  mean(vector <= value) * 100
}

similarity_features <- c("bat_HR", "bat_H", "bat_OPS", "pit_W", "pit_SV", "pit_ERA", "award_total", "allstar_games")

nearest_players <- function(target_row, pool, n = 10) {
  cols <- similarity_features[similarity_features %in% names(pool)]
  if (length(cols) == 0 || nrow(pool) == 0) return(data.frame())

  mat <- pool[, cols, drop = FALSE]
  target <- target_row[, cols, drop = FALSE]

  for (c in cols) {
    mat[[c]] <- suppressWarnings(as.numeric(mat[[c]]))
    target[[c]] <- suppressWarnings(as.numeric(target[[c]]))
    med <- median(mat[[c]], na.rm = TRUE)
    if (is.na(med)) med <- 0
    mat[[c]][is.na(mat[[c]])] <- med
    target[[c]][is.na(target[[c]])] <- med
    sd_val <- sd(mat[[c]], na.rm = TRUE)
    if (is.na(sd_val) || sd_val == 0) sd_val <- 1
    mat[[c]] <- (mat[[c]] - med) / sd_val
    target[[c]] <- (target[[c]] - med) / sd_val
  }

  diffs <- sweep(as.matrix(mat), 2, as.numeric(target[1, cols]), FUN = "-")
  dist <- sqrt(rowSums(diffs^2))
  pool$similarity_distance <- dist
  pool <- pool[order(pool$similarity_distance), ]
  head(pool, n)
}

custom_nearest <- function(input_values, role_filter, n = 10) {
  pool <- players
  if (!is.null(role_filter) && role_filter != "All") {
    pool <- pool[pool$primary_role == role_filter, ]
  }
  if (nrow(pool) == 0) pool <- players

  target <- data.frame(
    bat_HR = input_values$hr,
    bat_H = input_values$hits,
    bat_OPS = input_values$ops,
    pit_W = input_values$wins,
    pit_SV = input_values$saves,
    pit_ERA = input_values$era,
    award_total = input_values$awards,
    allstar_games = input_values$allstars
  )
  nearest_players(target, pool, n = n)
}

# -----------------------------
# UI
# -----------------------------
ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      body { background-color: #f7f8fb; }
      .title-block { padding: 18px 22px; background: white; border-radius: 14px; margin-bottom: 18px; box-shadow: 0 1px 8px rgba(0,0,0,0.08); }
      .subtitle { color: #5a6475; font-size: 15px; margin-top: 4px; }
      .metric-card { background: white; padding: 16px; border-radius: 14px; box-shadow: 0 1px 8px rgba(0,0,0,0.08); margin-bottom: 12px; min-height: 96px; }
      .metric-label { color: #5a6475; font-size: 13px; text-transform: uppercase; letter-spacing: .04em; }
      .metric-value { font-size: 26px; font-weight: 700; margin-top: 5px; }
      .small-note { color: #6b7280; font-size: 13px; }
      .section-card { background: white; padding: 18px; border-radius: 14px; box-shadow: 0 1px 8px rgba(0,0,0,0.08); margin-bottom: 16px; }
      table { font-size: 13px; }
      .warning-note { background: #fff8e6; padding: 10px 14px; border-radius: 10px; margin-bottom: 12px; }
    "))
  ),

  div(
    class = "title-block",
    h1("Cooperstown Calculator"),
    div(class = "subtitle", "Interactive Hall of Fame prediction dashboard using the final Project 4 model outputs")
  ),

  tabsetPanel(
    id = "main_tabs",

    tabPanel(
      "Overview",
      br(),
      fluidRow(
        column(3, div(class = "metric-card", div(class = "metric-label", "Eligible players"), div(class = "metric-value", format(nrow(players), big.mark = ",")))),
        column(3, div(class = "metric-card", div(class = "metric-label", "Actual inductees"), div(class = "metric-value", format(sum(players$inducted == 1, na.rm = TRUE), big.mark = ",")))),
        column(3, div(class = "metric-card", div(class = "metric-label", "Best model"), div(class = "metric-value", ifelse(nrow(model_cmp) > 0, model_cmp$model[1], "NA")))),
        column(3, div(class = "metric-card", div(class = "metric-label", "Best PR AUC"), div(class = "metric-value", ifelse(nrow(model_cmp) > 0, round(model_cmp$pr_auc[1], 3), "NA"))))
      ),
      fluidRow(
        column(
          7,
          div(class = "section-card",
              h3("Model leaderboard"),
              p(class = "small-note", "The leaderboard is read directly from reports/tables/model_comparison_holdout.csv."),
              tableOutput("overview_model_table"),
              plotOutput("overview_model_plot", height = "280px")
          )
        ),
        column(
          5,
          div(class = "section-card",
              h3("Project workflow"),
              tags$ol(
                tags$li("Build a career-level player table from Lahman/SABR, Kaggle backup tables, pybaseball samples, and a BeautifulSoup scrape."),
                tags$li("Use EDA and unsupervised archetypes to understand different player profiles."),
                tags$li("Train supervised models and select the final model using rare-event metrics such as PR AUC."),
                tags$li("Export predictions into this app for user-friendly exploration.")
              )
          ),
          div(class = "section-card",
              h3("Source coverage"),
              tableOutput("source_table_short")
          )
        )
      )
    ),

    tabPanel(
      "Player Explorer",
      br(),
      sidebarLayout(
        sidebarPanel(
          selectizeInput("player", "Search or choose a player", choices = NULL, options = list(placeholder = "Type a player name...")),
          sliderInput("similar_n", "Similar players to show", min = 5, max = 20, value = 10),
          div(class = "warning-note", "Similarity is based on career statistics, awards, All-Star appearances, and pitching/batting profile. It is an explanation aid, not a second model.")
        ),
        mainPanel(
          fluidRow(
            column(4, div(class = "metric-card", div(class = "metric-label", "Predicted HoF probability"), div(class = "metric-value", textOutput("selected_probability")))),
            column(4, div(class = "metric-card", div(class = "metric-label", "Actual status"), div(class = "metric-value", textOutput("selected_actual")))),
            column(4, div(class = "metric-card", div(class = "metric-label", "Archetype"), div(class = "metric-value", textOutput("selected_archetype"))))
          ),
          div(class = "section-card", h3(textOutput("selected_title")), tableOutput("selected_profile_table")),
          fluidRow(
            column(6, div(class = "section-card", h3("Career profile percentiles"), plotOutput("selected_percentile_plot", height = "320px"))),
            column(6, div(class = "section-card", h3("Most similar historical players"), tableOutput("selected_similar_table")))
          )
        )
      )
    ),

    tabPanel(
      "Rankings",
      br(),
      sidebarLayout(
        sidebarPanel(
          selectInput("rank_role", "Role", choices = role_choices),
          selectInput("rank_cluster", "Archetype", choices = cluster_choices),
          selectInput("rank_status", "Actual status", choices = status_choices),
          sliderInput("rank_prob", "Predicted probability range", min = 0, max = 1, value = c(0, 1), step = 0.01),
          sliderInput("rank_n", "Rows to display", min = 10, max = 100, value = 25)
        ),
        mainPanel(
          div(class = "section-card", h3("Filtered candidate ranking"), tableOutput("ranking_table")),
          div(class = "section-card", h3("Probability distribution after filters"), plotOutput("ranking_hist", height = "260px"))
        )
      )
    ),

    tabPanel(
      "Archetypes",
      br(),
      sidebarLayout(
        sidebarPanel(
          selectInput("cluster_choice", "Choose an archetype", choices = cluster_choices),
          sliderInput("cluster_top_n", "Top players to show", min = 5, max = 30, value = 10)
        ),
        mainPanel(
          div(class = "section-card", h3("Archetype profile"), tableOutput("cluster_profile_table")),
          fluidRow(
            column(6, div(class = "section-card", h3("Top players in archetype"), tableOutput("cluster_top_table"))),
            column(6, div(class = "section-card", h3("Prediction spread"), plotOutput("cluster_probability_plot", height = "300px")))
          ),
          div(class = "section-card", h3("PCA archetype figure"), imageOutput("cluster_pca_image"))
        )
      )
    ),

    tabPanel(
      "What-if Similarity",
      br(),
      sidebarLayout(
        sidebarPanel(
          selectInput("custom_role", "Compare against role", choices = role_choices),
          numericInput("custom_hr", "Home runs", value = 250, min = 0),
          numericInput("custom_hits", "Hits", value = 1800, min = 0),
          numericInput("custom_ops", "OPS", value = 0.800, min = 0, step = 0.01),
          numericInput("custom_wins", "Pitching wins", value = 0, min = 0),
          numericInput("custom_saves", "Saves", value = 0, min = 0),
          numericInput("custom_era", "ERA", value = 3.50, min = 0, step = 0.1),
          numericInput("custom_awards", "Awards", value = 3, min = 0),
          numericInput("custom_allstars", "All-Star games", value = 3, min = 0),
          sliderInput("custom_n", "Nearest players", min = 5, max = 25, value = 10)
        ),
        mainPanel(
          fluidRow(
            column(6, div(class = "metric-card", div(class = "metric-label", "Nearest-neighbor average probability"), div(class = "metric-value", textOutput("custom_avg_prob")))),
            column(6, div(class = "metric-card", div(class = "metric-label", "Nearest-neighbor inducted rate"), div(class = "metric-value", textOutput("custom_inducted_rate"))))
          ),
          div(class = "section-card",
              h3("Closest historical profiles"),
              p(class = "small-note", "This tab does not call the Python model live. It finds similar historical players and summarizes their model probabilities."),
              tableOutput("custom_similar_table")
          )
        )
      )
    ),

    tabPanel(
      "Model & Methods",
      br(),
      fluidRow(
        column(6, div(class = "section-card", h3("Model comparison"), tableOutput("methods_model_table"))),
        column(6, div(class = "section-card", h3("Top feature importance"), tableOutput("feature_importance_table")))
      ),
      fluidRow(
        column(6, div(class = "section-card", h3("Precision-recall curve"), imageOutput("pr_curve_image"))),
        column(6, div(class = "section-card", h3("Calibration curve"), imageOutput("calibration_image")))
      ),
      div(class = "section-card",
          h3("Interpretation"),
          p("This dashboard is a communication layer for the final Project 4 pipeline. It does not replace the Python training pipeline; it reads the final prediction and evaluation outputs produced by the scripts."),
          p("The selected model emphasizes both statistical production and historical recognition. Career totals and peak metrics capture on-field production, while awards and All-Star appearances capture historical recognition.")
      )
    )
  )
)

# -----------------------------
# Server
# -----------------------------
server <- function(input, output, session) {
  updateSelectizeInput(
    session,
    "player",
    choices = players$display_name,
    selected = players$display_name[1],
    server = TRUE
  )

  selected_player <- reactive({
    idx <- match(input$player, players$display_name)
    if (is.na(idx)) idx <- 1
    players[idx, ]
  })

  filtered_rankings <- reactive({
    df <- players
    if (!is.null(input$rank_role) && input$rank_role != "All") df <- df[df$primary_role == input$rank_role, ]
    if (!is.null(input$rank_cluster) && input$rank_cluster != "All") df <- df[df$cluster_label == input$rank_cluster, ]
    if (!is.null(input$rank_status) && input$rank_status == "Inducted") df <- df[df$inducted == 1, ]
    if (!is.null(input$rank_status) && input$rank_status == "Not inducted") df <- df[df$inducted == 0, ]
    df <- df[df$predicted_probability >= input$rank_prob[1] & df$predicted_probability <= input$rank_prob[2], ]
    df <- df[order(-df$predicted_probability), ]
    df
  })

  output$overview_model_table <- renderTable({
    if (nrow(model_cmp) == 0) return(data.frame(Message = "Model comparison table not found."))
    out <- model_cmp[, intersect(c("model", "pr_auc", "roc_auc", "precision", "recall", "f1", "brier_score", "top_25_precision", "top_50_precision"), names(model_cmp)), drop = FALSE]
    for (c in setdiff(names(out), "model")) out[[c]] <- round(out[[c]], 3)
    out
  })

  output$overview_model_plot <- renderPlot({
    if (nrow(model_cmp) == 0 || !("pr_auc" %in% names(model_cmp))) return(NULL)
    df <- model_cmp
    df$model <- factor(df$model, levels = df$model[order(df$pr_auc)])
    ggplot(df, aes(x = model, y = pr_auc)) +
      geom_col() +
      coord_flip() +
      ylim(0, 1) +
      labs(x = "", y = "PR AUC", title = "Holdout PR AUC by model") +
      theme_minimal()
  })

  output$source_table_short <- renderTable({
    if (nrow(source_manifest) == 0) return(data.frame(Message = "Source manifest not found."))
    cols <- intersect(c("table", "source_used", "rows", "min_year", "max_year"), names(source_manifest))
    head(source_manifest[, cols, drop = FALSE], 8)
  })

  output$selected_title <- renderText({ selected_player()$player_name })
  output$selected_probability <- renderText({ format_probability(selected_player()$predicted_probability) })
  output$selected_actual <- renderText({ ifelse(selected_player()$inducted == 1, "Inducted", "Not inducted") })
  output$selected_archetype <- renderText({ selected_player()$cluster_label })

  output$selected_profile_table <- renderTable({
    p <- selected_player()
    data.frame(
      Field = c("Player ID", "Role", "Primary position", "Debut era", "Career years", "Eligibility year", "Cluster / archetype", "Actual Hall of Fame status"),
      Value = c(
        p$playerID,
        p$primary_role,
        p$primary_position,
        p$debut_era,
        paste0(format_number(p$debut_year), "–", format_number(p$final_year)),
        format_number(p$eligibility_year),
        p$cluster_label,
        ifelse(p$inducted == 1, "Inducted", "Not inducted")
      ),
      stringsAsFactors = FALSE
    )
  })

  output$selected_percentile_plot <- renderPlot({
    p <- selected_player()
    rows <- data.frame(
      Metric = unname(metric_labels[metric_cols]),
      Percentile = sapply(metric_cols, function(c) percentile_rank(p[[c]], players[[c]])),
      stringsAsFactors = FALSE
    )
    rows <- rows[!is.na(rows$Percentile), ]
    rows$Metric <- factor(rows$Metric, levels = rows$Metric[order(rows$Percentile)])
    ggplot(rows, aes(x = Metric, y = Percentile)) +
      geom_col() +
      coord_flip() +
      ylim(0, 100) +
      labs(x = "", y = "Percentile among eligible players", title = "Career profile percentiles") +
      theme_minimal()
  })

  output$selected_similar_table <- renderTable({
    p <- selected_player()
    pool <- players[players$playerID != p$playerID, ]
    if (!is.na(p$cluster_label)) pool <- pool[pool$cluster_label == p$cluster_label, ]
    if (nrow(pool) < input$similar_n) pool <- players[players$playerID != p$playerID, ]
    sim <- nearest_players(p, pool, n = input$similar_n)
    make_display_table(sim)[, c("Player", "Role", "Archetype", "Probability", "Actual", "HR", "Hits", "OPS", "Wins", "Saves", "Awards", "AllStar")]
  })

  output$ranking_table <- renderTable({
    df <- head(filtered_rankings(), input$rank_n)
    make_display_table(df)[, c("Player", "Role", "Position", "Archetype", "Probability", "Actual", "HR", "Hits", "OPS", "Wins", "Saves", "Awards", "AllStar")]
  })

  output$ranking_hist <- renderPlot({
    df <- filtered_rankings()
    if (nrow(df) == 0) return(NULL)
    ggplot(df, aes(x = predicted_probability)) +
      geom_histogram(bins = 30) +
      labs(x = "Predicted probability", y = "Players", title = "Prediction distribution") +
      theme_minimal()
  })

  output$cluster_profile_table <- renderTable({
    if (nrow(cluster_profiles) == 0) return(data.frame(Message = "Cluster profile table not found."))
    df <- cluster_profiles
    if (!is.null(input$cluster_choice) && input$cluster_choice != "All" && "cluster_label" %in% names(df)) {
      df <- df[df$cluster_label == input$cluster_choice, ]
    }
    df
  })

  output$cluster_top_table <- renderTable({
    df <- players
    if (!is.null(input$cluster_choice) && input$cluster_choice != "All") {
      df <- df[df$cluster_label == input$cluster_choice, ]
    }
    df <- df[order(-df$predicted_probability), ]
    out <- make_display_table(head(df, input$cluster_top_n))
    out[, c("Player", "Role", "Position", "Probability", "Actual", "HR", "Hits", "OPS", "Wins", "Saves", "Awards", "AllStar")]
  })

  output$cluster_probability_plot <- renderPlot({
    df <- players
    if (!is.null(input$cluster_choice) && input$cluster_choice != "All") {
      df <- df[df$cluster_label == input$cluster_choice, ]
    }
    if (nrow(df) == 0) return(NULL)
    ggplot(df, aes(x = predicted_probability)) +
      geom_histogram(bins = 25) +
      labs(x = "Predicted probability", y = "Players", title = "Predicted probability within archetype") +
      theme_minimal()
  })

  output$cluster_pca_image <- renderImage({
    p <- file.path(PROJECT_ROOT, "reports", "figures", "08_cluster_pca.png")
    if (!file.exists(p)) return(NULL)
    list(src = p, contentType = "image/png", width = "100%")
  }, deleteFile = FALSE)

  custom_similar <- reactive({
    values <- list(
      hr = input$custom_hr,
      hits = input$custom_hits,
      ops = input$custom_ops,
      wins = input$custom_wins,
      saves = input$custom_saves,
      era = input$custom_era,
      awards = input$custom_awards,
      allstars = input$custom_allstars
    )
    custom_nearest(values, input$custom_role, n = input$custom_n)
  })

  output$custom_avg_prob <- renderText({
    sim <- custom_similar()
    if (nrow(sim) == 0) return("NA")
    format_probability(mean(sim$predicted_probability, na.rm = TRUE))
  })

  output$custom_inducted_rate <- renderText({
    sim <- custom_similar()
    if (nrow(sim) == 0) return("NA")
    format_probability(mean(sim$inducted == 1, na.rm = TRUE))
  })

  output$custom_similar_table <- renderTable({
    sim <- custom_similar()
    out <- make_display_table(sim)
    out[, c("Player", "Role", "Position", "Archetype", "Probability", "Actual", "HR", "Hits", "OPS", "Wins", "Saves", "Awards", "AllStar")]
  })

  output$methods_model_table <- renderTable({
    if (nrow(model_cmp) == 0) return(data.frame(Message = "Model comparison table not found."))
    out <- model_cmp[, intersect(c("model", "pr_auc", "roc_auc", "precision", "recall", "f1", "balanced_accuracy", "brier_score", "top_25_precision", "top_50_precision"), names(model_cmp)), drop = FALSE]
    for (c in setdiff(names(out), "model")) out[[c]] <- round(out[[c]], 3)
    out
  })

  output$feature_importance_table <- renderTable({
    if (nrow(feature_imp) == 0) return(data.frame(Message = "Feature importance table not found."))
    imp_col <- if ("model_native_importance" %in% names(feature_imp)) "model_native_importance" else names(feature_imp)[2]
    out <- feature_imp[1:min(15, nrow(feature_imp)), c("feature", imp_col), drop = FALSE]
    names(out) <- c("Feature", "Importance")
    out$Feature <- clean_feature_name(out$Feature)
    out$Importance <- round(out$Importance, 4)
    out
  })

  output$pr_curve_image <- renderImage({
    p <- file.path(PROJECT_ROOT, "reports", "figures", "best_holdout_pr_curve.png")
    if (!file.exists(p)) return(NULL)
    list(src = p, contentType = "image/png", width = "100%")
  }, deleteFile = FALSE)

  output$calibration_image <- renderImage({
    p <- file.path(PROJECT_ROOT, "reports", "figures", "best_holdout_calibration_curve.png")
    if (!file.exists(p)) return(NULL)
    list(src = p, contentType = "image/png", width = "100%")
  }, deleteFile = FALSE)
}

shinyApp(ui = ui, server = server)
