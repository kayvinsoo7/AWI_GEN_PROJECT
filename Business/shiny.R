# Install necessary packages
install.packages("shiny")
install.packages("shap")

library(shiny)
library(shap)

# Load your trained model and data
# load("your_model.RData")
# load("your_data.RData")

# Define the UI for the app
ui <- fluidPage(
    titlePanel("Human-Machine Interface for AI - Counterfactual Analysis"),
    
    sidebarLayout(
        sidebarPanel(
            selectInput("feature", "Select Feature to Change:", 
                        choices = colnames(X_test)),
            numericInput("value", "Enter New Value for the Feature:", 0),
            actionButton("update", "Update Prediction"),
            hr(),
            verbatimTextOutput("prediction"),
            verbatimTextOutput("cf_prediction")
        ),
        
        mainPanel(
            plotOutput("shap_plot"),
            plotOutput("cf_shap_plot")
        )
    )
)

# Define server logic
server <- function(input, output) {
    
    # Reactive value to store modified data
    modified_data <- reactiveVal(X_test[1, , drop = FALSE])
    
    # Update the modified data when the button is clicked
    observeEvent(input$update, {
        new_data <- modified_data()
        new_data[, input$feature] <- input$value
        modified_data(new_data)
    })
    
    # Generate the initial prediction and SHAP values
    output$prediction <- renderPrint({
        predict(model, X_test[1, , drop = FALSE])
    })
    
    output$shap_plot <- renderPlot({
        shap_values <- shap::shap.values(xgb_model = model, X_train = X_train)
        shap::shap.plot.summary(shap_values)
    })
    
    # Generate the counterfactual prediction and SHAP values
    output$cf_prediction <- renderPrint({
        predict(model, modified_data())
    })
    
    output$cf_shap_plot <- renderPlot({
        shap_values_cf <- shap::shap.values(xgb_model = model, X_train = modified_data())
        shap::shap.plot.summary(shap_values_cf)
    })
}

# Run the application 
shinyApp(ui = ui, server = server)
