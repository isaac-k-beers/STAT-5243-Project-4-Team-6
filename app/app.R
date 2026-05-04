# Cooperstown Calculator — R Shiny demo
# Run after Python training creates app/model_predictions.csv

library(shiny)
library(ggplot2)

# Locate predictions CSV. Shiny runs from the app directory by default,
# so a plain relative path is the most portable choice. If the user runs
# this from elsewhere, fall back to a project-root-relative path.
pred_path <- "model_predictions.csv"
if (!file.exists(pred_path)) {
  alt_path <- file.path("app", "model_predictions.csv")
  if (file.exists(alt_path)) {
    pred_path <- alt_path
  } else {
    stop("model_predictions.csv not found. Run python scripts/03_train_models.py first.")
  }
}
players <- read.csv(pred_path, stringsAsFactors = FALSE)
players$display_name <- paste(players$player_name, "(", players$primary_role, ")", sep = "")

ui <- fluidPage(
  titlePanel("Cooperstown Calculator: Hall of Fame Predictor"),
  sidebarLayout(
    sidebarPanel(
      selectInput("player", "Choose a player:", choices = players$display_name),
      checkboxInput("show_top", "Show top predicted candidates", TRUE),
      helpText("This app uses precomputed predictions from the Python model pipeline.")
    ),
    mainPanel(
      h3(textOutput("selected_name")),
      h2(textOutput("probability")),
      tableOutput("player_table"),
      plotOutput("prob_plot"),
      conditionalPanel(
        condition = "input.show_top == true",
        h3("Top predicted eligible players"),
        tableOutput("top_table")
      )
    )
  )
)

server <- function(input, output) {
  selected <- reactive({
    players[players$display_name == input$player, ][1, ]
  })
  output$selected_name <- renderText({ selected()$player_name })
  output$probability <- renderText({
    paste0("Predicted induction probability: ", round(100 * selected()$predicted_probability, 1), "%")
  })
  output$player_table <- renderTable({
    p <- selected()
    data.frame(
      Role = p$primary_role,
      Primary_Position = p$primary_position,
      Debut_Era = p$debut_era,
      Archetype = p$cluster_label,
      Actual_Inducted = ifelse(p$inducted == 1, "Yes", "No"),
      HR = p$bat_HR,
      Hits = p$bat_H,
      OPS = round(p$bat_OPS, 3),
      Wins = p$pit_W,
      Saves = p$pit_SV,
      ERA = round(p$pit_ERA, 2),
      Awards = p$award_total,
      AllStar_Games = p$allstar_games
    )
  })
  output$prob_plot <- renderPlot({
    p <- selected()
    df <- data.frame(label = c("Probability", "Remaining"), value = c(p$predicted_probability, 1 - p$predicted_probability))
    ggplot(df, aes(x = label, y = value)) +
      geom_col() +
      ylim(0, 1) +
      labs(title = paste("Prediction for", p$player_name), x = "", y = "Probability") +
      theme_minimal()
  })
  output$top_table <- renderTable({
    players[order(-players$predicted_probability), c("player_name", "primary_role", "cluster_label", "predicted_probability", "inducted")][1:20, ]
  })
}

shinyApp(ui = ui, server = server)
