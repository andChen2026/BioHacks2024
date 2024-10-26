
#---------------------------------------------------------------------------
# EDA of the breast cancer data
# download source: clinicaltrials.gov
#---------------------------------------------------------------------------
library(here)
library(tidyverse)
library(ggtext)
#---------------------------------------------------------------------------
#  data load
#---------------------------------------------------------------------------
data <- read.csv("data/breast_cancer_all_clinicaltrials.csv")
str(data)
describe(data)
tibble::glimpse(data)
#---------------------------------------------------------------------------
#  graphing
#---------------------------------------------------------------------------
ggplot(data, aes(x = Conditions)) +
  geom_bar(color = "firebrick", stat="count") +
  #labs(x = "Year", y = "Temperature (°F)") +
  theme(plot.background = element_rect(fill = "gray60"),
        plot.margin = margin(t = 1, r = 3, b = 1, l = 8, unit = "cm"))+
  theme_minimal( base_size = 15)+
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    axis.ticks = element_blank(),
    axis.text.x = element_text(family = "Roboto Mono"),
    axis.title.x = element_text(margin = margin(t = 10),
                                size = 16),
    plot.title = element_markdown(face = "bold", size = 21),
    plot.subtitle = element_text(
      color = "grey40", hjust = 0,
      margin = margin(0, 0, 20, 0)
    ),
    plot.title.position = "plot",
    plot.caption = element_markdown(
      color = "grey40", lineheight = 1.2,
      margin = margin(20, 0, 0, 0)),
    plot.margin = margin(15, 15, 10, 15)
  )
