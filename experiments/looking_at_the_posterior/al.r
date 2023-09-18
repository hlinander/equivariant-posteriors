library(ggplot2)
library(RPostgreSQL)

drv <- dbDriver("PostgreSQL")
con <-
  dbConnect(
    drv,
    dbname = "equiv",
    host = "localhost",
    port = "5432",
    user = "postgres",
    password = "postgres"
  )

# Fetch data from my_table
query <- "SELECT * FROM active_learning;"
df <- dbGetQuery(con, query)

# Close the database connection
dbDisconnect(con)

p <- (ggplot(df, aes(x = fraction, y = value, group = model, color=model)) + 
        geom_line() + 
        facet_wrap( ~metric, scales="free_y") + 
        theme_minimal(base_size = 24))
print(p)
