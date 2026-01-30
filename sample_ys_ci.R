library(tidyverse)
library(tidymodels)

set.seed(424)
mys <- 42 * 1.1

sdys = 6
pop <- tibble(s = rnorm(1e5, mean = mys, sd = sdys))

# pop <- tibble(s= extRemes::revd(1e5,loc =1.09545113, scale =0.08230480, shape =-0.03560832))*mys
# replicate samples of size 10, repeated 100 times
samps <-
  rep_slice_sample(
    .data = pop,
    replace = FALSE,
    reps = 100,
    n = 10
  ) %>%
  summarise(
    ms = mean(s),
    lci = t.test(s, mu = mys/1.1)$conf.int[1],
    uci = t.test(s, mu = mys/1.1)$conf.int[2],
    yn = ifelse(between(mys/1.1, lci, uci), "Yes", "No")
  ) %>% 
  mutate(id = 1:n())

samps %>% count(ms < mys/1.1)
samps %>% count(between(ms, mys - 0.5 * sdys, mys + 0.5 * sdys))
samps %>% count(yn) # count of samples with mean ys/1.1 between uci and lci

samps %>%
  arrange(uci) %>% 
  mutate(yn = factor(yn, levels = c("Yes", "No"))) %>%
  ggplot(aes(id, ms, col = yn)) +
  geom_errorbar(
    aes(ymin = lci,
        ymax = uci),
    position = "dodge",
    width = 1,
    linewidth = 0.75
  ) +
  geom_point(aes(y = ms), col = 'blue',fill='blue', shape = 23) +
  scale_color_manual(values = c("red","grey80")) +
  geom_hline(yintercept = 42,
             lty = 2,
             linewidth = 0.75,
             col = 'orangered') +
  # geom_hline(yintercept = 42,
  #            lty = 2,
  #            linewidth = 0.75,
  #            col = 'black') +
  theme_bw(14) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank()) +
  labs(x = NULL,
       col = "Includes\nMean",
       y = "Yield Strength (ksi)")

